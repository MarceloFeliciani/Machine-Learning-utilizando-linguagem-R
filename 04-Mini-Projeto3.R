# Prevendo a inadimplência de clientes com Machine Learning e Power BI

# Definindo a pasta de trabalho
setwd("C:/Power_BI_2_DSA/Cap15-MachineLearning")
getwd()

# Definição do problema 
# Verificar se o cliente pagará ou não o cartão de crédito no próximo mês

# Instalando os pacotes
install.packages("Amelia")  # trata valores ausentes
install.packages('caret') # constroi modelos de Machine Learning e processa os dados
install.packages('ggplot2') # constroi gráficos
install.packages('dplyr') # manipula dados
install.packages('reshape') # muda o formato dos dados
install.packages('randomForest') # para o Machine Learning
install.packages('e1071') # para o Machine Learning

# carregando os pacotes
library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)

# carregando o dataset
# Fonte: https://archive.ics.uci.edu/ml/datesets/default+of+credit+card+clients
dados_clientes <- read.csv("05-dataset.csv")

# visualizando os dados e sua estrutura
View(dados_clientes)  # ver os dados e formato de tabela
dim(dados_clientes)  # ver as dimensões das linhas e colunas
str(dados_clientes) # ver os tipos das variáveis
summary(dados_clientes) # resumo estatístico


##################### Análise Exploratória, Limpeza e Transformação ##########

# Removendo a primeira coluna ID
dados_clientes$ID <- NULL   # passando a coluna ID para nulo, por que são apenas dados identificadores
dim(dados_clientes)  # mostra uma coluna a menos na dimensão coluna
View(dados_clientes) # mostra os dados em forma de panilha, sem o ID


# Renomeando a coluna de classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "Inadimplente"  # renomeando a coluna 24
colnames(dados_clientes)
View(dados_clientes)


# verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))  # sapply "varre" todo dataset. Somará se tiver valores faltantes
?missmap
missmap(dados_clientes, main = "Valores Missing Observados") # mostra gráfica a mesma coisa acima, se existe valores vazios
dados_clientes <- na.omit(dados_clientes) # se tiver valores ausentes esse comando omit vai remover


# convertendo os atributos genero, escolaridade, estado civil e idade
# para fatores (categorias)
str(dados_clientes) # ver os tipos das variáveis

# Renomeando colunas categóricas
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)
View(dados_clientes)

# Convertendo Genero de 2 e 1 para Masculino e Feminino
View(dados_clientes$Genero)  # vendo somente a coluna gênero
str(dados_clientes$Genero) # tipo de dado da coluna gênero
summary(dados_clientes$Genero) # estatística da coluna gênero, mas de forma numérica que é errado para Feminino e Masculino
?cut # converte variável tipo numérica para o tipo fator. Converte o tipo e o valor da variável
dados_clientes$Genero <- cut(dados_clientes$Genero,
                      c(0,1,2),
                      labels = c("Masculino","Feminino"))

View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero) # estatística da coluna gênero, mostrando total de masculino e femino. Variável fator


# Convertendo Escolaridade
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                                   c(0,1,2,3,4),
                                   labels = c("Pos Graduado",
                                              "Graduado",
                                              "Ensino Médio",
                                              "Outros"))

View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade) # total dos dados


# Convertendo Estado Civil
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,
                                   c(-1,0,1,2,3),
                                   labels = c('Desconhecido',
                                              'Casado',
                                              'Solteiro',
                                              'Outro'))
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil) # total dos dados


# convertendo a variável para o tipo fator com faixa etária
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
hist(dados_clientes$Idade) # histograma com as idades
dados_clientes$Idade <- cut(dados_clientes$Idade,
                            c(0,30,50,100),
                            labels = c('Jovem',
                                       'Adulto',
                                       'Idoso'))

View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade) # total dos dados

View(dados_clientes)


# convertendo a variavel que indica pagamentos para o tipo fator
# vou usar o as factor, ao invés do cut, por que vou mudar apenas o tipo da variável e não o seu valor
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)


# dataset após as conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))  # mostrou que tem 345 dados ausentes na coluna Escolaridade
missmap(dados_clientes, main = "Valores Missing Observados") # mostrou que tem 345 dados ausentes na coluna Escolaridade em modo gráfico
dados_clientes <- na.omit(dados_clientes) # vou tratar os valores ausentes, removendo-os

sapply(dados_clientes, function(x) sum(is.na(x))) # mostrou não tem valores ausentes
missmap(dados_clientes, main = "Valores Missing Observados") # mostrou não tem valores ausentes
dim(dados_clientes)

View(dados_clientes)


# Alterando a variável inadimplente para o tipo fator. Ela aparece no dataset como 0 e 1 (Não/Sim)
str(dados_clientes$Inadimplente)  # mostra que está com tipo inteiro 0 e 1
colnames(dados_clientes) # ver o nomes das colunas
dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)  # retirando do tipo inteiro e convertendo para o tipo fator
str(dados_clientes$Inadimplente) # agora está como fator
View(dados_clientes)


# total de inadimplentes versus não inadimplentes
?table
table(dados_clientes$Inadimplente)  # mostra o total de registros de cada classe


# vejamos as porcentagens entre as classes
prop.table(table(dados_clientes$Inadimplente))  # temos muito mais pessoas adimplentes do que inadimplentes. Issa análise faz sentido na área financeira


# plot da distribuição usando ggplot2
qplot(Inadimplente, data = dados_clientes, geom = "bar") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# set seed (processo randômico/aleatório para divisão dos dados em treino e teste)
set.seed(12345)


# amostragem etratificada 
# seleciona as linhas de acordo com a variável inadimplente como strata
?createDataPartition  # funçao que faz SPLITTING dos dados (dividir/separar)

# a divisao será feita com a variavel inadimplente
# p = 0.75 porcentagem = (75%) dos dados que vão para treinamento. Restando 0.25 (25%) para teste.
# list = FALSE o resultado será uma matriz. list = TRUE o resultado será uma lista
indice <- createDataPartition(dados_clientes$Inadimplente, p = 0.75, list = FALSE)  # 75% para treino e organizando numa matriz
dim(indice) # mostra a dimensão da matriz. 22242 linhas e 1 coluna.



# definimos os dados de treinamento como subconjunto do conjunto de dados original 
# com números de indice de linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,] # [] indica que vou fatiar os dados. Indice são as linhas, espaço vazio são as colunas. Indico que quero somente as linhas e não quero colunas
dim(dados_treino) # dimensão mostrando 22242 linhas e 24 colunas
table(dados_treino$inadimplente) 


# percentagem entre as classes
prop.table(table(dados_treino$Inadimplente)) # vendo a proporção que deve ser a mesma dos dados originais


# número de registros no dataset de treino
dim(dados_treino)


# comparamos as percentagens entre as classes de treinamento e dos dados originais
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)),
                       prop.table(table(dados_clientes$Inadimplente)))

colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# plot para ver a distribuição do treinameto versus original
ggplot(melt_compara_dados, aes(x = X1, y = value)) +
  geom_bar( aes(fill = X2), stat = "identity", position = 'dodge') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
# o gráfico mostrou que a proporção continua entre os dados treinamento e original


# tudo o que não está no dataset de treinamento está no dataset de teste. Observe o sinal - (menos)
dados_teste <- dados_clientes[-indice,]  # vou filtrar o meu dataset original, por tudo que NÃO estiver no índice
dim[dados_teste]
dim[dados_treino]



############################## MODELO MACHINE LEARNING ###################


# CONSTRUINDO A 1a VERSÃO DO MODELO
?randomForest  # conjunto de árvore de decisão (constrói modelos de classificação e regressão)
# classificação prevê uma classe/categoria.
# regressão prevê um valor numérico
View(dados_treino)
modelo_v1 <- randomForest(Inadimplente ~ ., data = dados_treino) # inadimplente é a variável, ~ representa uma fórmula. Ponto (.) todas as variáveis preditoras
modelo_v1 # após a criação do modelo. Mostrou que é um tipo de classificação, foram criada 500 árvores


# Avaliando o modelo
plot(modelo_v1)

# Previsões com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste) # usar dados de teste, por que os dados de treino ele já conhece

# confusion matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente, positive = '1')  # positive = 1, clientes que vão pagar
cm_v1  # teve acurácia de 81%. Está exelente. A partir de 70% já é rasoável
# abaixo de 50% não é aceitável
# de 50 a 70%, tentar melhor


# calculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1


# Balanceamento de classe
install.packages('DMwR')
library(DMwR)
?SMOTE

# Aplicando o SMOTE - SMOTE: Synthetic Minority Over-sampling Technique
# https://arxiv.org./pdf/1106.1813.pdf
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente)) # mostra que inadimplentes é menor do que adimplentes
# classe 0 pessoas que pagam, classe 1 inadimplentes.

set.seed(9560) # executa um processo randomico
dados_treino_bal <- SMOTE(Inadimplente ~ ., data = dados_treino) # balanceando. Fazendo os inadimpletes aumentares em quantidade proporcionalmente para igualar em quantidade com as pessoas que pagam corretamente
table(dados_treino_bal$inadimplente)
prop.table(table(dados_treino_bal$inadimplente)) # os dados estão muito mais próximos, equilibrados


# CONSTRUINDO A SEGUNDA VERSÃO DO MODELO
modelo_v2 <- randomForest(Inadimplente ~ ., data = dados_treino_bal)
modelo_v2

# Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste) # usar dados de teste, por que os dados de treino ele já conhece

# confusion matrix
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inadimplente, positive = '1')  # positive = 1, clientes que vão pagar
cm_v2  # teve acurácia de 79%. Está exelente. A partir de 70% já é rasoável. Tem um equilibrio maior do que o primeiro
# abaixo de 50% não é aceitável
# de 50 a 70%, tentar melhor


# calculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1
# o modelo 1, acerteva mais a classe 1 do que 0.
# este modelo 2, acerta mais as 2 classes, está usando os dados balanceados



# Importancia das variáveis preditoras para as previões
imp_var <- importance(modelo_v2)
varImportance <- data.frame(variables = row.names(imp_var),
                            importance = round(imp_var[,"MeanDecreaseGini"],2))

# Criando o rank de variáveis baseado na importância
rankImportance <- varImportance %>%
  mutate(Rank = paste0("#", dense_rank(desc(importance))))

# usando ggplot2 para visualizar a importância relativa das variáveis 
ggplot(rankImportance,
       aes(x = reorder(variables, importance),
           y = importance,
           fill = importance)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(x = variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()

         
       
# CONSTRUINDO A TERCEIRA VERSÃO DO MODELO APENA COM VARIÁVEIS MAIS IMPORTANTES
colnames(dados_treino_bal)
# concatenando as variaveis mais importantes
modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          data = dados_treino_bal)
modelo_v3


# Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste) # usar dados de teste, por que os dados de treino ele já conhece

# confusion matrix
?caret::confusionMatrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inadimplente, positive = '1')  # positive = 1, clientes que vão pagar
cm_v3  # teve acurácia de 79%. Está exelente. A partir de 70% já é rasoável
# abaixo de 50% não é aceitável
# de 50 a 70%, tentar melhor


# calculando Precision, Recall e F1-score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1
# o terceiro modelo ficou ainda melhor do que o 2o modelo por que usei apaena as variáveis mais importantes



# SALVANDO O MODELO EM DISCO
# enquanto não for salvo o modelo estará na memória RAM do computador, podendo ser perdido
# já ter criado o diretório modelo
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# CARREGANDO O MODELO
modelo_final <- readRDS('modelo/modelo_v3.rds')


# Previsões com dados de 3 clientes
# Dados dos 3 clientes
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

# concatena em um data frame
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)


# previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(novos_clientes)
# reclamou do tipo de dado

# checando os tipos de dados
str(dados_treino_bal)
str(dados_clientes)

# convertendo os tipos de dados para factor para ficar igual ao modelo
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

# previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_clientes) # o cliente ID 1 ficará inadimplente (classe 1)


##### FIM






