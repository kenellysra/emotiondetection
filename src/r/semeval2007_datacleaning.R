#Set Directory
setwd("/home/kenelly/workspaces/emotionnewsheadlines/emotiondetection/resources")

#Reading emotions file and including colnames
# <- read.table("affectivetext_testemotionsgold", header = FALSE, sep = " ") 
emotions <- read.table("affectivetext_trialemotionsgold", header = FALSE, sep = " ")
emotions[,1] <- NULL

colnames(emotions) <- c("anger", "disgust", "fear", "joy", "sadness", "surprise")
head(emotions)
#Converting News Headlines from XML to Data Frame
library(XML)
library(data.table)
#newsheadlinesxml <- xmlParse("affectivetext_test.xml") 
newsheadlinesxml <- xmlParse("affectivetext_trial.xml")
newsheadlineslist <- xmlToList(newsheadlinesxml)

newsheadlineslist <- lapply(newsheadlineslist, function(x) as.data.table(as.list(x)))
newsheadlines <- rbindlist(newsheadlineslist, use.names = TRUE, fill = T)
newsheadlines <- newsheadlines[-nrow(newsheadlines),]

newsemotions <- cbind(newsheadlines[,1:2], emotions)
colnames(newsemotions)[1:2] <- c("news_headlines", "origin_ID")
newsemotions <- newsemotions[,c("origin_ID", "news_headlines", "anger", "disgust", "fear", "joy", "sadness", "surprise")]

write.csv(newsemotions, file = "semeval2007datatrial.csv")

