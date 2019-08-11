library("ggplot2")
data<- read.csv(file="data/titles.csv")
score<- data$total[which(data$total < 201)]
print(score[20:60])


hist(score)
