# Detecção Probabilística de Fraude com VIB, Contraste e Predição Conforme Indutiva

**Autor:** Luiz Tiago Wilcke  
**Linguagem:** Python / PyTorch  
**Tópicos:** Variational Information Bottleneck (VIB), InfoNCE Contrastive Loss, Focal Loss, Split Conformal Prediction, Detecção de Anomalias

---

## 1. Visão Geral

Este repositório contém a implementação em PyTorch de um pipeline para detecção de transações fraudulentas e anomalias financeiras sob condições de severo desbalanceamento de classes.

A abordagem combina quatro componentes:
1. **Compressão e Regularização Estocástica (VIB):** Um codificador variacional que mapeia atributos tabulares de alta dimensão em uma representação latente comprimida e estocástica $z \sim q_\phi(z\vert{}x)$.
2. **Aprendizado Auto-Supervisionado Contrastivo (InfoNCE):** Projeção latente contrastiva para garantir separabilidade topológica de representações sob perturbações estocásticas.
3. **Função de Perda Focal Ponderada:** Otimização discriminativa com foco em exemplos difíceis para mitigar desbalanceamento severo (ex.: $<2\%$ de fraudes).
4. **Quantificação de Incerteza Livre de Distribuição (Split Conformal Prediction):** Garantia teórica de cobertura marginal $1 - \alpha$ para os conjuntos de predição gerados, identificando transações ambíguas (onde o conjunto predito é $\{0, 1\}$).

---

## 2. Fundamentação Teórica e Formulação Matemática

### 2.1. Variational Information Bottleneck (VIB)

O objetivo do VIB é encontrar uma representação latente intermediária $Z$ que maximize a informação mútua com o rótulo $Y$, limitando a quantidade de informação retida da entrada $X$:

$$\max_\phi I(Z; Y) - \beta I(Z; X)$$

Pela cota variacional inferior, a perda de regularização latente (divergência KL com prior gaussiano padrão $\mathcal{N}(0, I)$) é calculada analiticamente por:

$$D_{\text{KL}}\left(q_\phi(z\vert{}x) \parallel \mathcal{N}(0, I)\right) = -\frac{1}{2} \sum_{j=1}^{d_z} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

onde $\mu(x)$ e $\log \sigma^2(x)$ são as saídas do codificador e a amostragem é feita via truque de reparametrização:

$$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

### 2.2. Função de Perda Focal Binária (Class Imbalance)

Para penalizar transações fraudulentas raras sem saturação por negativos fáceis, utiliza-se a Focal Loss com fator modulador $\gamma$:

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

onde:
$$p_t = \begin{cases} \hat{p}, & \text{se } y = 1 \\ 1 - \hat{p}, & \text{se } y = 0 \end{cases} \quad \text{e} \quad \alpha_t = \begin{cases} \alpha, & \text{se } y = 1 \\ 1 - \alpha, & \text{se } y = 0 \end{cases}$$

No código: $\alpha = 0.80$ e $\gamma = 2.0$.

---

### 2.3. Perda Contrastiva InfoNCE

Dadas duas vistas aumentadas de um mesmo lote transacional $x^{(1)} = x + \delta_1$ e $x^{(2)} = x + \delta_2$ com projeções normalizadas $z_i^{(1)}, z_i^{(2)} \in \mathbb{S}^{d_p-1}$, a similaridade de cosseno com temperatura $\tau$ é definida por:

$$\text{sim}(u, v) = \frac{u^\top v}{\Vert{}u\Vert{}_2 \Vert{}v\Vert{}_2}$$

A perda InfoNCE para o par positivo $(i, j)$ no minibatch de tamanho $2B$ é formulada como:

$$\mathcal{L}_{\text{InfoNCE}} = -\sum_{i=1}^{2B} \log \frac{\exp\left(\text{sim}(z_i, z_{p(i)}) / \tau\right)}{\sum_{k \neq i}^{2B} \exp\left(\text{sim}(z_i, z_k) / \tau\right)}$$

---

### 2.4. Função de Perda Total Multi-Objetivo

O modelo é treinado de ponta a ponta minimizando a combinação ponderada:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Focal}}(\hat{y}, y) + \beta \mathcal{L}_{\text{KL}}(z) + \lambda_{\text{rec}} \mathcal{L}_{\text{MSE}}(\hat{x}, x) + \lambda_{\text{cont}} \mathcal{L}_{\text{InfoNCE}}(z^{(1)}, z^{(2)})$$

onde:
* $\beta = 10^{-3}$ (compressão do gargalo variacional)
* $\lambda_{\text{rec}} = 0.1$ (reconstrução do espaço de atributos)
* $\lambda_{\text{cont}} = 0.05$ (separabilidade de representação latente)

---

### 2.5. Predição Conforme Indutiva (Split Conformal)

Para contornar probabilidades mal calibradas em redes neurais profundas, aplica-se Conformal Prediction indutivo com nível de significância $\alpha \in (0, 1)$ (garantindo $1 - \alpha = 95\%$ de cobertura marginal teórica).

#### Score de Não-Conformidade:
Para uma amostra de calibração $(x_i, y_i)$, o score quantifica o erro da probabilidade prevista na classe verdadeira:

$$s_i = 1 - \hat{P}(Y = y_i \mid x_i)$$

#### Quantil Empírico com Correção para Amostras Finitas:
Dado um conjunto de calibração $\mathcal{D}_{\text{calib}}$ de tamanho $m$, define-se o quantil $\hat{q}$ com correção conservadora:

$$\hat{q} = \text{Quantil}\left( \{s_i\}_{i=1}^m ; \; \frac{\lceil(m+1)(1-\alpha)\rceil}{m} \right)$$

#### Conjunto de Predição na Inferência:
Para uma nova transação $x_{n+1}$, o conjunto predito $C(x_{n+1}) \subseteq \{0, 1\}$ é:

$$C(x_{n+1}) = \left\{ y \in \{0, 1\} : 1 - \hat{P}(Y = y \mid x_{n+1}) \le \hat{q} \right\}$$

Propriedade de cobertura garantida:
$$P\left( Y_{n+1} \in C(X_{n+1}) \right) \ge 1 - \alpha$$

---

## 3. Arquitetura da Rede

```text
Entrada (x in R^28)
       |
  [Codificador VIB] -> MLP(28 -> 64 -> 32) + BatchNorm + LeakyReLU
       |
       +---> Media: mu(x) in R^10
       +---> Log-Variancia: log_var(x) in R^10
       |
 [Reparametrizacao] -> z = mu + sigma * epsilon in R^10
       |
       +--------------------+---------------------+
       |                    |                     |
[Classificador VIB]  [Decodificador REC]   [Projetor InfoNCE]
MLP(10 -> 16 -> 1)   MLP(10 -> 32 -> 64 -> 28) MLP(10 -> 16 -> 8)
    Sigmoid               MSE Loss            InfoNCE Loss
       |
       v
 Prob. Fraude P(Y=1|X)
       |
       v
 [Calibrador Conforme] -> Conjunto de Predicao C(X) in {{0}, {1}, {0, 1}}
