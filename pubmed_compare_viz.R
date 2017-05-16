library(tidyr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(gridExtra)

# Data import
df_lda = read.csv('pubmed results/pubmed_lda.csv') %>%
  select(2:4) %>%
  mutate(method='LDA')

df_nmf = read.csv('pubmed results/pubmed_nmf.csv') %>%
  select(2:4) %>%
  mutate(method='NMF')

df_tsnmf = read.csv('pubmed results/pubmed_tsnmf.csv') %>%
  select(2:6)

df_tsnmf_01 <- df_tsnmf %>%
  filter(supervision==0.01) %>%
  select(one_of(c('avg_similarity','n_topics_resolved','rep'))) %>%
  mutate(method='TS-NMF 0.01')

df_tsnmf_10 <- df_tsnmf %>%
  filter(supervision==0.10) %>%
  select(one_of(c('avg_similarity','n_topics_resolved','rep'))) %>%
  mutate(method='TS-NMF 0.1')

df_tsnmf_20 <- df_tsnmf %>%
  filter(supervision==0.20) %>%
  select(one_of(c('avg_similarity','n_topics_resolved','rep'))) %>%
  mutate(method='TS-NMF 0.2')

df_tsnmf_50 <- df_tsnmf %>%
  filter(supervision==0.50) %>%
  select(one_of(c('avg_similarity','n_topics_resolved','rep'))) %>%
  mutate(method='TS-NMF 0.5')

df_tsnmf_80 <- df_tsnmf %>%
  filter(supervision==0.80) %>%
  select(one_of(c('avg_similarity','n_topics_resolved','rep'))) %>%
  mutate(method='TS-NMF 0.8')
  
df <- rbind(df_lda,df_nmf,df_tsnmf_01, df_tsnmf_10, df_tsnmf_20,df_tsnmf_50,df_tsnmf_80) %>%
  gather('ScoreType', 'Score',1:2) %>%
  mutate(ScoreType = ifelse(ScoreType=="avg_similarity", "Average Weighted Jaccard", "# of Topics Resolved"))

# Helpful plotting functions and data
number_ticks <- function(n) {
  function(limits) {
    pretty(limits, n)
  }
}

# Plots

textsize <- 8

p1 <- df %>%
  ggplot(aes(x=method, y=Score)) +
  facet_wrap(~ScoreType, scales="free") +
  geom_jitter(size=0.75, width=0.005, alpha=.5) +
  geom_boxplot(aes(group=method), 
               position = "identity",
               alpha=0.0,
               outlier.shape=NA,
               lwd=.25) +
  theme_minimal(base_size = textsize) +
  labs(x='', y='') +
  scale_y_continuous(breaks=number_ticks(10)) + 
  theme(strip.text.y = element_text(size = textsize),
        axis.text=element_text(size=textsize),
        axis.text.x = element_text(angle = 45, hjust = 1))
p1

ggsave('results/comparemodels_pubmed.png', p1)
