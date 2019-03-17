#install.packages("chron")
#install.packages("ggplot2")
#install.packages("caret")
#install.packages("data.table")
#install.packages("mltools")
#install.packages("dummies")
#install.packages("zoo")

library(caret)
library(sqldf)
library(dummies)
library(ggplot2)
library(zoo)
library(chron)
library(data.table)
library(mltools)
library(e1071)

##############Data Preparation############################################
#Trips_2018_Q1 = read.csv("Divvy_Trips_2018_Q1.csv")
#Trips_2018_Q2 = read.csv("Divvy_Trips_2018_Q2.csv")
#Trips_2018_Q3 = read.csv("Divvy_Trips_2018_Q3.csv")
#Trips_2018_Q4 = read.csv("Divvy_Trips_2018_Q4.csv")
#colnames(Trips_2018_Q1) <- colnames(Trips_2018_Q2)
#Trips_2018 = rbind(Trips_2018_Q1,Trips_2018_Q2,Trips_2018_Q3,Trips_2018_Q4)
#write.csv(Trips_2018,"Trips_2018.csv")

Trips_2018 = read.csv("Trips_2018.csv")

#Change datetime column to separate date and time
Trips_2018$Start_Time <- format(as.POSIXct(Trips_2018$start_time,format="%Y-%m-%d %H:%M:%S"),"%H:%M:%S")
Trips_2018$Start_Date <- format(as.POSIXct(Trips_2018$start_time,format="%Y-%m-%d %H:%M:%S"),"%m/%d/%Y")
Trips_2018$start_time <- NULL

#Separate hour from time and make separate column for it to help in analysis
Trips_2018$start_hour =  as.numeric(substr(Trips_2018$Start_Time, 1, 2))

#Extract weekday and month from Start_Date to help in analysis
Trips_2018$day <- weekdays(Trips_2018$Start_Date)
Trips_2018$month <- months(Trips_2018$Start_Date)

#weather data for 365 days
weatherdata = read.csv("weather_2018.csv")
weatherdata$Start_Date = gsub("-","/",weatherdata$date)
weatherdata$Start_Date = strptime(as.character(weatherdata$Start_Date), "%Y/%m/%d")
weatherdata$Start_Date = format(weatherdata$Start_Date,"%m/%d/%Y")
weatherdata$date <- NULL
summary(weatherdata)
weatherdata$percipitation <- as.numeric(as.character(weatherdata$percipitation))
weatherdata$new_snow <- as.numeric(as.character(weatherdata$new_snow))
for (j in 1:ncol(weatherdata)){ 
  weatherdata[is.na(weatherdata[, j]), j] <- mean(weatherdata[, j], na.rm = TRUE) 
  }

#merging weather data and selected station's data
tripsData = merge(x = Trips_2018, y = weatherdata[ , c("Start_Date","average","percipitation","new_snow","snow_depth")], by.x = "Start_Date", by.y = "Start_Date", all.y=TRUE)
summary(tripsData)
#write.csv(tripsData,"tripsData.csv")
tripsData = read.csv("tripsData.csv")

##############exploratory analysis - 16 October incident############################################
hfilter <- c("8") # the impacted hour

dfilter <- c("09/11/2018","09/18/2018","09/25/2018",
             "10/02/2018","10/09/2018","10/16/2018","10/23/2018","10/30/2018")

tuesdayData = tripsData[tripsData$Start_Date %in% dfilter
                         & tripsData$start_hour %in% hfilter
                         ,]
summary(tuesdayData)
graphdata = sqldf("select Start_Date, COUNT(*) from tuesdayData GROUP BY Start_Date")
colnames(graphdata) <- c("Start_Date","Trips_Count")
ggplot(data=graphdata, aes(x=Start_Date, y=Trips_Count, group=1)) +
geom_line() +
geom_point()

##############exploratory analysis - for modeling############################################
dayData = sqldf("Select Start_Date,day,count(*) from tripsData GROUP BY Start_Date")
colnames(dayData)[3] <- "count"
ggplot(data=dayData, aes(x=Start_Date, y=count, group=day, color=day))+
  geom_line()+
  theme(axis.text.x = element_text(angle=90,hjust=1))

dayData = sqldf("Select Start_Date,day,count(*) from tripsData WHERE month='Dec' GROUP BY Start_Date")
colnames(dayData)[3] <- "count"
ggplot(data=dayData, aes(x=Start_Date, y=count, group=day, color=day))+
  geom_line()+
  theme(axis.text.x = element_text(angle=90,hjust=1))

###############Predict no of bikes needed at one station#####################################
stationsCount = sqldf("Select from_station_id,count(*) from tripsData GROUP BY from_station_id")
stationData = tripsData[tripsData$from_station_id==35,]

##data for each hour for 365 days
modeldata = sqldf("Select month,day,start_hour,average,percipitation,new_snow,snow_depth,count(*) from stationData GROUP BY from_station_id, Start_Date,start_hour")
colnames(modeldata)[8] <- "count"
summary(modeldata)

#one-hot encoding
modeldata$start_hour = factor(modeldata$start_hour)
model_data_encoded <- dummy.data.frame(modeldata, names=c("day","month","start_hour"), sep="_")

#Spliting training and test sets in 75\% and 25\% portions.
set.seed(101) # Setting Seed so that same sample is reproducible
sample <- sample.int(n = nrow(model_data_encoded), size = floor(.75*nrow(model_data_encoded)), replace = F)
training_set <- model_data_encoded[sample, ]
test_set  <- model_data_encoded[-sample, ]
#separating label/count from the features
training_set_y <- training_set$count
training_set_X <- training_set[,names(training_set) != "count"]
test_set_y <- test_set$count
test_set_X <- test_set[,names(test_set) != "count"]

######################Training####################################
#linear regression training
b = paste(colnames(training_set_X),collapse = "+")
training_set_names = paste("count ~ ",b,sep = "")

mlrfit=lm(training_set_names, data=training_set)

#SVR model training
svmfit = svm(training_set_y~., data = training_set_X, kernel = "radial") #cost=1, gamma=0.0212766, epsilon=0.1
svmfit = svm(training_set_y~., data = training_set_X, kernel = "polynomial") #cost=1, gamma=0.0212766, epsilon=0.1, degree=3

######################Evaluation####################################
#MLR
predictions = predict.lm(mlrfit, test_set_X)
RMSE = sqrt(mean(predictions - test_set_y)^2)

#SVR
predicted <- predict(svmfit, test_set_X)
RMSE = sqrt(mean(predicted - test_set_y)^2)

#tune the model and evaluate again
rbf.tune = tune.svm(X,y, kernel="radial",
                    gamma=c(0.01,0.02,0.03,0.04),
                    epsilon=c(0.0,0.1,0.2,0.3),
                    cost = c(1,2,3,4))
svmfit = svm(y~., data = X, kernel = "radial",
             gamma=rbf.tune$best.parameters$gamma, 
             epsilon=rbf.tune$best.parameters$epsilon,
             cost=rbf.tune$best.parameters$cost
             )

#tune the model and evaluate again
svmfit = svm(training_set_y~., data = training_set_X, kernel = "polynomial",
             gamma=rbf.tune$best.parameters$gamma, 
             epsilon=rbf.tune$best.parameters$epsilon,
             cost=rbf.tune$best.parameters$cost,
             degree=poly.tune$best.parameters$degree
            )
poly.tune = tune.svm(training_set_X,training_set_y, kernel="polynomial",
                     gamma=c(0.01,0.02,0.03,0.04),
                     epsilon=c(0.0,0.1,0.2,0.3),
                     cost = c(1,2,3,4),
                     degree=c(1,2,3,4,5))

