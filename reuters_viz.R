library(tidyr)
library(dplyr)
library(ggplot2)
library(ggthemes)

df_w_E = read.csv('results/data/reuters/reuters_wsnmf_by_supervision_rate_1.csv') %>%
  select(2:6) %>%
  mutate(Error = 'Weighted')

df_wo_E = read.csv('results/data/reuters/reuters_wsnmf_by_supervision_rate_no_E.csv') %>%
  select(2:6) %>%
  mutate(Error = 'Unweighted')

df <- rbind(df_w_E, df_wo_E) %>%
  gather('ScoreType', 'Score',1:2) %>%
  mutate(ScoreType = ifelse(ScoreType=="avg_similarity", "Average Weighted Jaccard", "# of Topics Resolved"))

df_30 <- read.csv('results/data/reuters/reuters_wsnmf_large.csv') %>%
  select(2:6) %>%
  gather('ScoreType', 'Score',1:2) %>%
  mutate(ScoreType = ifelse(ScoreType=="avg_similarity", "Average Weighted Jaccard", "# of Topics Resolved"))

p1 <- df %>%
  ggplot(aes(x=supervision, y=Score, color=Error)) +
  facet_grid(ScoreType ~ ., scales="free") +
  geom_point() +
  geom_smooth(method='loess', se = FALSE) +
  theme_minimal(base_size = 12) +
  labs(x='Supervision Rate', y='Score') +
  scale_colour_brewer(palette = "Set1")
  
p1

ggsave('results/data/reuters/errorweighting.png',p1)

number_ticks <- function(n) {
  function(limits) {
    pretty(limits, n)
  }
}

dummy <- data.frame(supervision=c(0,1),
                    Score=c(0,.24,0,90),
                    ScoreType=c("Average Weighted Jaccard","Average Weighted Jaccard",
                                "# of Topics Resolved","# of Topics Resolved"))

p2 <- df_30 %>%
  ggplot(aes(x=supervision, y=Score)) +
  facet_grid(ScoreType ~ ., scales="free") +
  stat_summary(fun.y=mean, geom="line", aes(group=1), 
               color='steelblue3', size=1) +
  geom_jitter(size=1, width=0.005, alpha=.5) +
  geom_boxplot(aes(group=supervision), 
               width=0.06, 
               position = "identity",
               alpha=0.0,
               outlier.shape=NA) +
  theme_minimal(base_size = 12) +
  geom_blank(data=dummy) +
  labs(x='Supervision Rate', y='Score') +
  scale_x_continuous(breaks=seq(0,1,0.1)) +
  scale_y_continuous(breaks=number_ticks(10))
p2

ggsave('results/data/reuters/score_by_supervision.png',p2)

p3 <- df_30 %>% 
  filter(ScoreType == "# of Topics Resolved") %>%
  ggplot(aes(x=supervision, y=topic_coverage)) +
  geom_point(alpha=0.5) +
  stat_summary(fun.y=mean, geom="line", aes(group=1), 
               color='black', size=0.75) +
  theme_minimal(base_size = 12) +
  scale_x_continuous(breaks=seq(0,1,0.1)) +
  scale_y_continuous(breaks=seq(0,1,0.1)) +
  labs(x='Supervision Rate', y='Topic Coverage')
p3

ggsave('results/data/reuters/coverage_by_supervision.png',p3)

p4 <- df_30 %>%
  filter(ScoreType == "Average Weighted Jaccard") %>%
  filter(supervision > 0.0) %>%
  group_by(supervision) %>%
  mutate(score_dev_from_group_mean = Score-mean(Score)) %>%
  mutate(coverage_dev_from_group_mean = topic_coverage-mean(topic_coverage)) %>%
  ggplot(aes(x=coverage_dev_from_group_mean, y=score_dev_from_group_mean))+
  geom_point() +
  theme_minimal(base_size = 12) +
  labs(x='Topic Coverage Deviation from Mean', y='Score Deviation from Mean')

p4

ggsave('results/data/reuters/score_by_coverage.png',p4)

