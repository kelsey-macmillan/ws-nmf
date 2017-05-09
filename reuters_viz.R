library(tidyr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(gridExtra)

# Data import
df_w_E = read.csv('results/reuters_wsnmf_by_supervision_rate.csv') %>%
  select(2:6) %>%
  mutate(Error = 'Weighted')

df_wo_E = read.csv('results/reuters_wsnmf_by_supervision_rate_no_E.csv') %>%
  select(2:6) %>%
  mutate(Error = 'Unweighted')

df_error <- rbind(df_w_E, df_wo_E) %>%
  gather('ScoreType', 'Score',1:2) %>%
  mutate(ScoreType = ifelse(ScoreType=="avg_similarity", "Average Weighted Jaccard", "# of Topics Resolved"))

df_30 <- read.csv('results/reuters_wsnmf_large.csv') %>%
  select(2:6) %>%
  gather('ScoreType', 'Score',1:2) %>%
  mutate(ScoreType = ifelse(ScoreType=="avg_similarity", "Average Weighted Jaccard", "# of Topics Resolved"))

# Helpful plotting functions and data
number_ticks <- function(n) {
  function(limits) {
    pretty(limits, n)
  }
}

dummy <- data.frame(supervision=c(0,1),
                    Score=c(0,.24,0,90),
                    ScoreType=c("Average Weighted Jaccard","Average Weighted Jaccard",
                                "# of Topics Resolved","# of Topics Resolved"),
                    Error = c("Weighted","Weighted","Weighted","Weighted"))


whisker_high <- function(y){
  iqr = quantile(y, probs=0.75) - quantile(y, probs=0.25)
  min(quantile(y, probs=0.75)+1.5*iqr, max(y))
}

whisker_low <- function(y){
  iqr = quantile(y, probs=0.75) - quantile(y, probs=0.25)
  max(quantile(y, probs=0.25)-1.5*iqr, min(y))
}

# Plots

textsize <- 8

p1 <- df_30 %>%
  ggplot(aes(x=supervision, y=Score)) +
  facet_wrap(~ScoreType, scales="free", ncol=1) +
  stat_summary(fun.y=mean, geom="line", aes(group=1), 
               color='steelblue3', size=0.5) +
  stat_summary(fun.y = whisker_high, geom = "line", aes(group=1), 
               linetype = "dashed", color='steelblue3') +
  stat_summary(fun.y = whisker_low, geom = "line", aes(group=1), 
               linetype = "dashed", color='steelblue3') +
  geom_jitter(size=0.75, width=0.005, alpha=.5) +
  geom_boxplot(aes(group=supervision), 
               width=0.06, 
               position = "identity",
               alpha=0.0,
               outlier.shape=NA,
               lwd=.25) +
  theme_minimal(base_size = textsize) +
  geom_blank(data=dummy) +
  labs(x='Supervision Rate', y='') +
  scale_x_continuous(breaks=seq(0,1,0.1)) +
  scale_y_continuous(breaks=number_ticks(10)) + 
  theme(strip.text.x = element_text(size = textsize),
        axis.text=element_text(size=textsize))
p1

p2 <- df_error %>%
  ggplot(aes(x=supervision, y=Score, color=Error)) +
  facet_wrap(~ScoreType, scales="free", ncol=1) +
  geom_point(size=0.75) +
  geom_blank(data=dummy) +
  geom_smooth(method='loess', se=FALSE, size=0.5) +
  theme_minimal(base_size = textsize) +
  labs(x='Supervision Rate', y='') +
  scale_color_manual(values = c("Weighted"="steelblue3","Unweighted"="tomato3")) +
  scale_x_continuous(breaks=seq(0,1,0.1)) +
  scale_y_continuous(breaks=number_ticks(10)) + 
  theme(legend.position = c(0.2, 0.9),
        legend.title = element_blank(),
        legend.key.height=unit(0.5,"cm"),
        legend.text = element_text(size = textsize),
        strip.text.x = element_text(size = textsize),
        axis.text=element_text(size=textsize))
  
p2

g <- grid.arrange(p1, p2, ncol=2, nrow=1)

g

ggsave('results/supervisionrate.png', g)