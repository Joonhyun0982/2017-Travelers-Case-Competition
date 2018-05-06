#=============================================================================
# 2017 Travelers Case Competition
#     Method: Other Machine Learning Methods
#=============================================================================


# ============== #
# 1. Preparation #
# ============== #

# install.packages('caret', dependencies = TRUE)
# install.packages("randomForest")
# install.packages("gbm")
# install.packages("neuralnet")

library("caret")  #folds creation
library("class")  #for KNN
library("rpart")  #for classification tree
library("randomForest")  #for random forest
library("gbm")  #for gbm
library("neuralnet")  #Neural Network
library("ROCR") #for ROC, AUC
library("mice")  #for missing data imputation on test set

train = read.csv("Train.csv")
head(train)

# Data cleaning
train <- subset(train, train$cancel!=-1)
train <- subset(train, train$ni.age < 100)
train <- subset(train, train$ni.age - train$len.at.res > 0)
train <- train[complete.cases(train[,1:18]),] #deleting rows with NA. Can be done by train <- na.omit(train) as well

# Constructing dummy variables for qualitative variables
train1 <- train
train1$ni.genderM <- ifelse(train1$ni.gender == "M", 1, 0)
train1$sales.channelBroker <- ifelse(train1$sales.channel == "Broker", 1, 0)
train1$sales.channelOnline <- ifelse(train1$sales.channel == "Online", 1, 0)
train1$sales.channelPhone <- ifelse(train1$sales.channel == "Phone", 1, 0)
train1$coverage.typeA <- ifelse(train1$coverage.type == "A", 1, 0)
train1$coverage.typeB <- ifelse(train1$coverage.type == "B", 1, 0)
train1$coverage.typeC <- ifelse(train1$coverage.type == "C", 1, 0)
train1$dwelling.typeCondo <- ifelse(train1$dwelling.type == "Condo", 1, 0)
train1$dwelling.typeHouse <- ifelse(train1$dwelling.type == "House", 1, 0)
train1$dwelling.typeTenant <- ifelse(train1$dwelling.type == "Tenant", 1, 0)
train1$creditlow <- ifelse(train1$credit == "low", 1, 0)
train1$creditmedium <- ifelse(train1$credit == "medium", 1, 0)
train1$credithigh <- ifelse(train1$credit == "high", 1, 0)
train1$house.colorblue <- ifelse(train1$house.color == "blue", 1, 0)
train1$house.colorred <- ifelse(train1$house.color == "red", 1, 0)
train1$house.colorwhite <- ifelse(train1$house.color == "white", 1, 0)
train1$house.coloryellow <- ifelse(train1$house.color == "yellow", 1, 0)
train1$zip.codeAZ <- ifelse(train1$zip.code >= 85000 & train1$zip.code < 86000, 1, 0)
train1$zip.codeCO <- ifelse(train1$zip.code >= 80000 & train1$zip.code < 81000, 1, 0)
train1$zip.codeDC <- ifelse(train1$zip.code >= 20000 & train1$zip.code < 21000, 1, 0)
train1$zip.codeIA <- ifelse(train1$zip.code >= 50000 & train1$zip.code < 51000, 1, 0)
train1$zip.codePA <- ifelse(train1$zip.code >= 15000 & train1$zip.code < 16000, 1, 0)
train1$zip.codeWA <- ifelse(train1$zip.code >= 98000 & train1$zip.code < 99000, 1, 0)

# Removing redundant variables
train1 <- subset(train1, select = -c(ni.gender, sales.channel, coverage.type, dwelling.type, credit, house.color, year, zip.code))

# data frame for auc of each methods
results <- data.frame(Method = as.numeric(), AUC = as.numeric())


# ================================== #
# 2. KNN along with cross validation #
# ================================== #

train2 <- subset(train1, select=-c(id, cancel))

# Convert the dependent var to factor. Normalize the numeric variables
train.cancel <- factor(train1$cancel)
ind <- sapply(train2, is.numeric)
train2[ind] <- lapply(train2[ind], scale)

# Creating folds randomly with equal size and no overlapping between folds
folds <- createFolds(train1$id, k = 10, list = TRUE, returnTrain = FALSE)

# KNN with k values from 1 to 29
accuracy <- data.frame(fold = as.numeric())
auc <- data.frame(fold = as.numeric())
for(i in 1:10) {
  k_fold_test <- train2[folds[[i]],]
  k_fold_train <- train2[-folds[[i]],]
  
  train.def <- train.cancel[-folds[[i]]]
  test.def <- train.cancel[folds[[i]]]
  
  for(j in 1:15) {
    knn <-  knn(k_fold_train, k_fold_test, train.def, k=2*j-1, prob=TRUE)
    accuracy[i,j+1] <- sum(test.def == knn)/nrow(k_fold_test)
    prob <- as.data.frame(knn)
    prob$pred <- ifelse(knn == 1, attr(knn,"prob"), 1-attr(knn,"prob"))
    auc[i,j+1] <- as.numeric(performance(prediction(prob$pred, test.def),"auc")@y.values)
  }
  accuracy[i,1] <- i
  auc[i,1] <- i
}
# changing variable names of accuracy/auc matrix
accuracy[i+1,1] <- "mean"
for(j in 1:15) {
  colnames(accuracy)[j+1] <- paste0("k",2*j-1)
  accuracy[i+1,j+1] <- mean(accuracy[-(i+1),j+1])
}
auc[i+1,1] <- "mean"
for(j in 1:15) {
  colnames(auc)[j+1] <- paste0("k",2*j-1)
  auc[i+1,j+1] <- mean(auc[-(i+1),j+1])
}

results[1,1] <- "KNN"
results[1,2] <- auc[11,"k25"]


# ======== #
# 3. Trees #
# ======== #

############### Classification Tree

modCT <- rpart(cancel ~ .-id, method="class", data=train1)

printcp(modCT) # display the results 
plotcp(modCT) # visualize cross-validation results 
summary(modCT) # detailed summary of splits

# plot tree 
plot(modCT, uniform=TRUE, main="Classification Tree for cancel")
text(modCT, use.n=TRUE, all=TRUE, cex=.8)
# ROC curve, AUC
pred <- prediction(predict(modCT, train1)[,2], train1$cancel)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(0,1)
performance(pred,"auc")

results[2,1] <- "Classification Tree"
results[2,2] <- performance(pred,"auc")@y.values

############### Regression Tree

modRT <- rpart(cancel ~ .-id, method="anova", data=train1)

printcp(modRT) # display the results 
plotcp(modRT) # visualize cross-validation results 
summary(modRT) # detailed summary of splits
rsq.rpart(modRT) # visualize cross-validation results

# plot tree 
plot(modRT, uniform=TRUE, main="Regression Tree for cancel")
text(modRT, use.n=TRUE, all=TRUE, cex=.8)
# ROC curve, AUC
pred <- prediction(predict(modRT, train1), train1$cancel)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(0,1)
performance(pred,"auc")

results[3,1] <- "Regression Tree"
results[3,2] <- performance(pred,"auc")@y.values

############### Random Forest

train1$cancel <- as.factor(train1$cancel)
modRF <- randomForest(cancel ~ .-id, data=train1)
summary(modRF)
getTree(modRF, k=2, labelVar=TRUE)

# ROC curve, AUC
pred <- prediction(modRF$votes[,2], train1$cancel)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(0,1)
performance(pred,"auc")

results[4,1] <- "Random Forest"
results[4,2] <- performance(pred,"auc")@y.values


# =========== #
# 4. Boosting #
# =========== #

train1$cancel <- train$cancel
modGBM <- gbm(cancel~.-id, data=train1, shrinkage=0.01, distribution = 'bernoulli', cv.folds=5, n.trees=3000, verbose=F)

# check the best iteration number
best.iter = gbm.perf(modGBM, method="cv")
best.iter
summary(modGBM)

for (i in 1:length(modGBM)) {
  plot.gbm(modGBM, i, best.iter)
}

# ROC curve, AUC
pred <- prediction(predict(modGBM, train1), train1$cancel)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(0,1)
performance(pred,"auc")

results[5,1] <- "Generalized Boosted Models"
results[5,2] <- performance(pred,"auc")@y.values


# ================== #
# 5. Neural Networks #
# ================== #

train2$cancel <- train1$cancel

n <- names(train2)
f <- as.formula(paste("cancel ~", paste(n[!n %in% "cancel"], collapse = " + ")))
modNN <- neuralnet(f, data=train2, hidden=c(3,2), stepmax = 1e+06, linear.output=FALSE)

plot(modNN)

# ROC curve, AUC
pred <- prediction(modNN$net.result, train2$cancel)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(0,1)
performance(pred,"auc")
results[6,1] <- "Neural Networks"
results[6,2] <- performance(pred,"auc")@y.values


# ====================== #
# 6. Logestic Regression #
# ====================== #

mod <- glm(cancel ~ creditlow +sales.channelBroker +creditmedium +zip.codeDC +n.children 
            +zip.codePA +claim.ind +ni.age +len.at.res +ni.marital.status +tenure +n.adults
            +zip.codeCO, family=binomial ,data=train1)
summary(mod)

# ROC curve, AUC
pred <- prediction(mod$fitted.values, train1$cancel)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(0,1)
performance(pred,"auc")
results[7,1] <- "Logestic Regression"
results[7,2] <- performance(pred,"auc")@y.values

results
