import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ==============================================================================
# 1. GERAÇÃO DE DADOS TRANSACIONAIS SINTÉTICOS (ALTO DESBALANCEAMENTO)
# ==============================================================================
def gerar_dados_transacoes(n_amostras=10000, n_atributos=30, taxa_fraude=0.015, semente=42):
    np.random.seed(semente)
    torch.manual_seed(semente)
    
    n_fraudes = int(n_amostras * taxa_fraude)
    n_licitas = n_amostras - n_fraudes
    
    # Transações lícitas: distribuições Gaussianas correlacionadas padrão
    atributos_licitos = np.random.randn(n_licitas, n_atributos) * 0.8
    rotulos_licitos = np.zeros(n_licitas, dtype=np.float32)
    
    # Transações fraudulentas / lavagem: anomalias multimodais com maior dispersão
    atributos_fraude_1 = np.random.randn(n_fraudes // 2, n_atributos) * 2.2 + 1.8
    atributos_fraude_2 = np.random.randn(n_fraudes - (n_fraudes // 2), n_atributos) * 1.5 - 2.5
    atributos_fraude = np.vstack([atributos_fraude_1, atributos_fraude_2])
    rotulos_fraude = np.ones(n_fraudes, dtype=np.float32)
    
    atributos_totais = np.vstack([atributos_licitos, atributos_fraude]).astype(np.float32)
    rotulos_totais = np.concatenate([rotulos_licitos, rotulos_fraude]).astype(np.float32)
    
    # Embaralhamento determinístico
    indices = np.random.permutation(n_amostras)
    atributos_totais = atributos_totais[indices]
    rotulos_totais = rotulos_totais[indices]
    
    return torch.tensor(atributos_totais), torch.tensor(rotulos_totais)


# ==============================================================================
# 2. FUNÇÕES DE PERDA ESPECÍFICAS (FOCAL LOSS E INFONCE)
# ==============================================================================
class PerdaFocalBinaria(nn.Module):
    def __init__(self, alfa=0.75, gama=2.0):
        super(PerdaFocalBinaria, self).__init__()
        self.alfa = alfa
        self.gama = gama

    def forward(self, probabilidades, rotulos):
        probabilidades = torch.clamp(probabilidades, 1e-7, 1.0 - 1e-7)
        pt = torch.where(rotulos == 1, probabilidades, 1.0 - probabilidades)
        fator_alfa = torch.where(rotulos == 1, self.alfa, 1.0 - self.alfa)
        fator_modulador = (1.0 - pt) ** self.gama
        perda = -fator_alfa * fator_modulador * torch.log(pt)
        return perda.mean()


class PerdaInfoNCE(nn.Module):
    def __init__(self, temperatura=0.1):
        super(PerdaInfoNCE, self).__init__()
        self.temperatura = temperatura

    def forward(self, projecao_1, projecao_2):
        lote_tam = projecao_1.shape[0]
        z1 = F.normalize(projecao_1, dim=1)
        z2 = F.normalize(projecao_2, dim=1)
        
        representacoes = torch.cat([z1, z2], dim=0)
        matriz_similaridade = torch.matmul(representacoes, representacoes.T) / self.temperatura
        
        mascara_auto_similaridade = torch.eye(2 * lote_tam, device=projecao_1.device).bool()
        matriz_similaridade.masked_fill_(mascara_auto_similaridade, -float('inf'))
        
        rotulos_positivos = torch.cat([
            torch.arange(lote_tam, 2 * lote_tam, device=projecao_1.device),
            torch.arange(0, lote_tam, device=projecao_1.device)
        ])
        
        return F.cross_entropy(matriz_similaridade, rotulos_positivos)


# ==============================================================================
# 3. ARQUITETURA DO MODELO PROBABILÍSTICO HÍBRIDO (VIB + REC + CONTRASTIVO)
# ==============================================================================
class ModeloDeteccaoFraudeVIB(nn.Module):
    def __init__(self, dim_entrada=30, dim_latente=12, dim_projecao=8):
        super(ModeloDeteccaoFraudeVIB, self).__init__()
        
        # 1. Codificador Estocástico Variacional
        self.espinha_dorsal_codificador = nn.Sequential(
            nn.Linear(dim_entrada, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )
        self.camada_media = nn.Linear(32, dim_latente)
        self.camada_log_variancia = nn.Linear(32, dim_latente)
        
        # 2. Decodificador Reconstrutor
        self.decodificador = nn.Sequential(
            nn.Linear(dim_latente, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, dim_entrada)
        )
        
        # 3. Classificador Discriminativo de Fraude (Saída do VIB)
        self.classificador = nn.Sequential(
            nn.Linear(dim_latente, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 4. Projetor Contrastivo
        self.projetor_contrastivo = nn.Sequential(
            nn.Linear(dim_latente, 16),
            nn.ReLU(),
            nn.Linear(16, dim_projecao)
        )

    def codificar(self, x):
        h = self.espinha_dorsal_codificador(x)
        media = self.camada_media(h)
        log_var = torch.clamp(self.camada_log_variancia(h), -10.0, 5.0)
        return media, log_var

    def reparametrizar(self, media, log_var):
        desvio_padrao = torch.exp(0.5 * log_var)
        ruido_epsilon = torch.randn_like(desvio_padrao)
        return media + desvio_padrao * ruido_epsilon

    def forward(self, x):
        media, log_var = self.codificar(x)
        z = self.reparametrizar(media, log_var)
        
        predicao_fraude = self.classificador(z).squeeze(-1)
        reconstrucao = self.decodificador(z)
        projecao = self.projetor_contrastivo(z)
        
        return predicao_fraude, reconstrucao, projecao, media, log_var


# ==============================================================================
# 4. CALIBRADOR CONFORME INDUTIVO (SPLIT CONFORMAL)
# ==============================================================================
class CalibradorConformeIndutivo:
    def __init__(self, nivel_significancia=0.05):
        self.alfa = nivel_significancia
        self.quantil_corte = None

    def calibrar(self, probabilidades_calib, rotulos_calib):
        m = len(rotulos_calib)
        
        # Score de não-conformidade: probabilidade atribuída à classe errada
        # s_i = 1 - p(y_real | x)
        probabilidade_classe_correta = torch.where(
            rotulos_calib == 1,
            probabilidades_calib,
            1.0 - probabilidades_calib
        ).detach().cpu().numpy()
        
        scores_nao_conformidade = 1.0 - probabilidade_classe_correta
        
        # Posição do quantil empírico finito com correção conservadora (m+1)/m
        posto_quantil = np.ceil((m + 1) * (1.0 - self.alfa)) / m
        posto_quantil = min(posto_quantil, 1.0)
        
        self.quantil_corte = np.percentile(scores_nao_conformidade, posto_quantil * 100)

    def predizer_conjunto(self, probabilidades_teste):
        probs = probabilidades_teste.detach().cpu().numpy()
        conjuntos_preditos = []
        
        for prob in probs:
            p_classe_0 = 1.0 - prob
            p_classe_1 = prob
            
            score_0 = 1.0 - p_classe_0
            score_1 = 1.0 - p_classe_1
            
            conjunto = []
            if score_0 <= self.quantil_corte:
                conjunto.append(0)
            if score_1 <= self.quantil_corte:
                conjunto.append(1)
            
            # Garantia de conjunto não-vazio
            if len(conjunto) == 0:
                conjunto = [int(p_classe_1 >= 0.5)]
                
            conjuntos_preditos.append(conjunto)
            
        return conjuntos_preditos


# ==============================================================================
# 5. PIPELINE DE TREINAMENTO E AVALIAÇÃO
# ==============================================================================
if __name__ == "__main__":
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carregamento e particionamento dos dados
    atributos, rotulos = gerar_dados_transacoes(n_amostras=12000, n_atributos=28, taxa_fraude=0.02)
    
    n_treino = 7000
    n_calib = 2500
    n_teste = 2500
    
    dados_treino = TensorDataset(atributos[:n_treino], rotulos[:n_treino])
    dados_calib = TensorDataset(atributos[n_treino:n_treino+n_calib], rotulos[n_treino:n_treino+n_calib])
    dados_teste = TensorDataset(atributos[n_treino+n_calib:], rotulos[n_treino+n_calib:])
    
    carregador_treino = DataLoader(dados_treino, batch_size=128, shuffle=True, drop_last=True)
    
    # 2. Inicialização do Modelo e Critérios
    modelo = ModeloDeteccaoFraudeVIB(dim_entrada=28, dim_latente=10, dim_projecao=8).to(dispositivo)
    otimizador = optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-4)
    
    criterio_focal = PerdaFocalBinaria(alfa=0.80, gama=2.0)
    criterio_infonce = PerdaInfoNCE(temperatura=0.1)
    criterio_reconstrucao = nn.MSELoss()
    
    peso_kl_beta = 1e-3
    peso_contrastivo = 0.05
    peso_rec = 0.1
    
    # 3. Loop de Treinamento
    n_epocas = 35
    modelo.train()
    for epoca in range(n_epocas):
        perda_acumulada = 0.0
        for lote_x, lote_y in carregador_treino:
            lote_x = lote_x.to(dispositivo)
            lote_y = lote_y.to(dispositivo)
            
            # Duas vistas estocásticas para contraste
            ruido_aumento_1 = lote_x + torch.randn_like(lote_x) * 0.05
            ruido_aumento_2 = lote_x + torch.randn_like(lote_x) * 0.05
            
            otimizador.zero_grad()
            
            # Passada 1
            pred_y, rec_x, proj_1, media, log_var = modelo(ruido_aumento_1)
            # Passada 2 (obter segunda projeção)
            _, _, proj_2, _, _ = modelo(ruido_aumento_2)
            
            # Cálculo dos termos de perda
            perda_focal = criterio_focal(pred_y, lote_y)
            perda_kl = -0.5 * torch.mean(torch.sum(1.0 + log_var - media.pow(2) - log_var.exp(), dim=1))
            perda_rec = criterio_reconstrucao(rec_x, lote_x)
            perda_cont = criterio_infonce(proj_1, proj_2)
            
            perda_total = perda_focal + (peso_kl_beta * perda_kl) + (peso_rec * perda_rec) + (peso_contrastivo * perda_cont)
            
            perda_total.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
            otimizador.step()
            
            perda_acumulada += perda_total.item()
            
        if (epoca + 1) % 10 == 0 or epoca == n_epocas - 1:
            print(f"Época {epoca+1:02d}/{n_epocas} | Perda Total Ponderada: {perda_acumulada / len(carregador_treino):.5f}")

    # 4. Calibração Conforme no Conjunto de Calibração
    modelo.eval()
    with torch.no_grad():
        x_calib, y_calib = dados_calib.tensors
        pred_prob_calib, _, _, _, _ = modelo(x_calib.to(dispositivo))
        
        calibrador = CalibradorConformeIndutivo(nivel_significancia=0.05) # 95% de cobertura teórica
        calibrador.calibrar(pred_prob_calib.cpu(), y_calib)

    # 5. Avaliação Rigorosa no Conjunto de Teste
    with torch.no_grad():
        x_teste, y_teste = dados_teste.tensors
        pred_prob_teste, rec_teste, _, _, _ = modelo(x_teste.to(dispositivo))
        
        conjuntos_preditos = calibrador.predizer_conjunto(pred_prob_teste.cpu())
        y_teste_np = y_teste.numpy()
        
        # Verificação de Cobertura Conforme Real
        coberturas = [y_teste_np[i] in conjuntos_preditos[i] for i in range(len(y_teste_np))]
        cobertura_empirica = np.mean(coberturas)
        
        tamanhos_conjuntos = [len(c) for c in conjuntos_preditos]
        tamanho_medio = np.mean(tamanhos_conjuntos)
        
        # Detecção de transações ambíguas (onde o conjunto de predição é {0, 1})
        indices_incertos = [i for i, c in enumerate(conjuntos_preditos) if len(c) == 2]
        
    print("\n" + "="*70)
    print("MÉTRICAS DO MODELO E PREVISÃO CONFORME (TESTE)")
    print("="*70)
    print(f"Nível de Cobertura Teórico (1 - alfa) : 95.00%")
    print(f"Cobertura Empírica Obtida em Teste   : {cobertura_empirica * 100:.2f}%")
    print(f"Tamanho Médio dos Conjuntos Conformes: {tamanho_medio:.4f}")
    print(f"Transações Críticas com Incerteza     : {len(indices_incertos)} de {n_teste} ({len(indices_incertos)/n_teste*100:.2f}%)")
    print(f"Limiar Conforme de Não-Conformidade  : q_hat = {calibrador.quantil_corte:.5f}")
    print("="*70)
    
    # Amostra de decisões para inspeção
    print("\nExemplo de Predições Conformes:")
    for i in range(5):
        print(f"Transação #{i+1:02d} | Rótulo Real: {int(y_teste_np[i])} | "
              f"Prob. Fraude: {pred_prob_teste[i].item():.4f} | "
              f"Conjunto Conforme: {conjuntos_preditos[i]}")