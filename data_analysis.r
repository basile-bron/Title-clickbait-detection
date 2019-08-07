library("dplyr")
library("ggplot2")
title<-read.csv(file="real_data/train/titles_uniq.txt", header=TRUE, sep=",")
scores <- read.csv(file="real_data/train/score_uniq.txt", header=TRUE, sep=",")

ggplot(scores, aes(x=X23)) +   geom_histogram(binwidth=4, fill="#69b3a2", color="#e9ecef", )


final<-data.frame(title$X10.Sad.Ways.Logan.Paul.s.Life.Has.Changed.After.the.Scandal[1:1288832],scores)
