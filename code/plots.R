library(ggplot2)

f <- read.csv(r"{C:\Users\leand\Documents\_Studie\MSc\Jaar 1\2. ML\ml-semester-project\data\dur.csv}", header=FALSE,
              stringsAsFactors = FALSE)
f[,1] <- as.numeric(f[,1])
ggplot(f, aes(as.factor(x=V1))) + 
  geom_bar(fill='black') + 
  xlab('Duration (samples)') + 
  theme_classic(base_size=10) + 
  geom_text(stat='count', aes(label=..count..), vjust=-.25, size=3) +
  scale_y_continuous('Count', labels=NULL)
  #scale_x_continuous("Duration", labels= as.character(seq(1,max(f$V1))), breaks=seq(1,max(f$V1)))
