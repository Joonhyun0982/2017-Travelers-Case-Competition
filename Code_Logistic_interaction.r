#=============================================================================
# 2017 Travelers Case Competition
#     Method: Logistic Regression
#=============================================================================


# ============== #
# 1. Preparation #
# ============== #

#install.packages("caret")
#install.packages("My.stepwise")
#install.packages("ResourceSelection")
#install.packages("ROCR")
#install.packages("mice")

library("caret")  #qplot, folds creation
library("My.stepwise")  #Stepwise selection
library("ResourceSelection")  #Hosmer-Lemeshow GOF statistic
library("ROCR")  #to construct ROC curve
library("mice")  #for missing data imputation on test set

train = read.csv("Train.csv")
head(train)

# Data cleaning
train <- subset(train, train$cancel!=-1)
train <- subset(train, train$ni.age < 100)
train <- subset(train, train$ni.age - train$len.at.res >= 0)
train <- train[complete.cases(train[,1:18]),] #deleting rows with NA. Can be done by train <- na.omit(train) as well

# Constructing dummy variables for qualitative variables
train1 <- train
train1$ni.genderM <- ifelse(train$ni.gender == "M", 1, 0)
train1$sales.channelBroker <- ifelse(train$sales.channel == "Broker", 1, 0)
train1$sales.channelOnline <- ifelse(train$sales.channel == "Online", 1, 0)
train1$sales.channelPhone <- ifelse(train$sales.channel == "Phone", 1, 0)
train1$coverage.typeA <- ifelse(train$coverage.type == "A", 1, 0)
train1$coverage.typeB <- ifelse(train$coverage.type == "B", 1, 0)
train1$coverage.typeC <- ifelse(train$coverage.type == "C", 1, 0)
train1$dwelling.typeCondo <- ifelse(train$dwelling.type == "Condo", 1, 0)
train1$dwelling.typeHouse <- ifelse(train$dwelling.type == "House", 1, 0)
train1$dwelling.typeTenant <- ifelse(train$dwelling.type == "Tenant", 1, 0)
train1$creditlow <- ifelse(train$credit == "low", 1, 0)
train1$creditmedium <- ifelse(train$credit == "medium", 1, 0)
train1$credithigh <- ifelse(train$credit == "high", 1, 0)
train1$house.colorblue <- ifelse(train$house.color == "blue", 1, 0)
train1$house.colorred <- ifelse(train$house.color == "red", 1, 0)
train1$house.colorwhite <- ifelse(train$house.color == "white", 1, 0)
train1$house.coloryellow <- ifelse(train$house.color == "yellow", 1, 0)
train1$zip.codeAZ <- ifelse(train$zip.code >= 85000 & train1$zip.code < 86000, 1, 0)
train1$zip.codeCO <- ifelse(train$zip.code >= 80000 & train1$zip.code < 81000, 1, 0)
train1$zip.codeDC <- ifelse(train$zip.code >= 20000 & train1$zip.code < 21000, 1, 0)
train1$zip.codeIA <- ifelse(train$zip.code >= 50000 & train1$zip.code < 51000, 1, 0)
train1$zip.codePA <- ifelse(train$zip.code >= 15000 & train1$zip.code < 16000, 1, 0)
train1$zip.codeWA <- ifelse(train$zip.code >= 98000 & train1$zip.code < 99000, 1, 0)


# ============================ #
# 2. Exploratory data analysis #
# ============================ #

# Tests for regrouping categorical variables
tst <- glm(cancel ~ sales.channelOnline +sales.channelPhone, family=binomial, data = train1)
summary(tst)
tst <- glm(cancel ~ sales.channelBroker +sales.channelPhone, family=binomial, data = train1)
summary(tst)
tst <- glm(cancel ~ sales.channelBroker +sales.channelOnline, family=binomial, data = train1)
summary(tst)  # We see sales.channelOnline and sales.channelPhone can be grouped
# Repeat these tests for all variables with 3 or more categories

# Reconstructing dummy variables after initial examination
train1$sales.channelNonBroker <- ifelse(train1$sales.channel != "Broker", 1, 0)

# Correlation matrix
train2 <- subset(train1, select = -c(id, ni.gender, sales.channel, sales.channelOnline, sales.channelPhone,
                                     coverage.type, dwelling.type, credit, house.color, year, zip.code))
train1.cov <- cor(train2)

# Interaction Plots
train2 <- subset(train, select = -c(id, zip.code, premium, len.at.res, ni.age, house.color, year, cancel))
for (i in 1:length(train2)) {
  for (j in 1:length(train2)) {
    interaction.plot(train2[[c(i)]], train2[[c(j)]], train$cancel, trace.label = names(train2)[j],
                     xlab = names(train2)[i], ylab = "Mean of cancel")
  }
}

# Plots to see whether there are interactions between continueous variable and categorical variables
qplot(x = n.children, y = cancel, facets = ~n.adults, data = train) + geom_smooth(method = "glm")
qplot(x = ni.age, y = cancel, facets = ~sales.channel, data = train) + geom_smooth(method = "glm")
qplot(x = ni.age, y = cancel, facets = ~coverage.type, data = train) + geom_smooth(method = "glm")
qplot(x = ni.age, y = cancel, facets = ~ni.marital.status, data = train) + geom_smooth(method = "glm")

# full model with all 2x2 interaction terms for exploration purpose
train2 <- subset(train, select = -c(id, year, zip.code))
mod0 <- glm(cancel ~ .*., family=binomial, data=train2)
options(max.print=10000)
summary(mod0)


# ========================== #
# 3. Model fitting/selection #
# ========================== #

# initial full model with possible interaction terms
mod0 <- glm(cancel ~ tenure +claim.ind +n.adults +n.children +ni.gender +ni.marital.status +premium
            +sales.channelNonBroker +coverage.type +dwelling.type +len.at.res +credit +house.color +ni.age
            +zip.codeAZ +zip.codeCO +zip.codeDC +zip.codeIA +zip.codePA +n.children*n.adults +ni.gender*ni.marital.status
            +coverage.type*dwelling.type +coverage.type*ni.age +sales.channelNonBroker*ni.age +ni.marital.status*ni.age
            +ni.marital.status*coverage.type, family=binomial, data=train1)
summary(mod0)

# Creating interaction variables after initial examination
train1$n.children.n.adults <- train1$n.children*train1$n.adults
train1$ni.genderM.ni.marital.status <- train1$ni.genderM*train1$ni.marital.status
train1$coverage.typeB.dwelling.typeHouse <- train1$coverage.typeB*train1$dwelling.typeHouse
train1$coverage.typeB.dwelling.typeTenant <- train1$coverage.typeB*train1$dwelling.typeTenant
train1$coverage.typeC.dwelling.typeHouse <- train1$coverage.typeC*train1$dwelling.typeHouse
train1$coverage.typeC.dwelling.typeTenant <- train1$coverage.typeC*train1$dwelling.typeTenant
train1$ni.marital.status.ni.age <- train1$ni.marital.status*train1$ni.age
train1$ni.marital.status.coverage.typeB <- train1$ni.marital.status*train1$coverage.typeB
train1$ni.marital.status.coverage.typeC <- train1$ni.marital.status*train1$coverage.typeC
train1$coverage.typeB.ni.age <- train1$coverage.typeB*train1$ni.age
train1$coverage.typeC.ni.age <- train1$coverage.typeC*train1$ni.age
train1$sales.channelNonBroker.ni.age <- train1$sales.channelNonBroker*train1$ni.age

# Stepwise regression while checking VIF simultaneously
my.variable.list <- c("tenure", "claim.ind", "n.adults", "n.children", "ni.genderM", "ni.marital.status", "premium",
                      "sales.channelNonBroker", "coverage.typeB", "coverage.typeC", "dwelling.typeHouse",
                      "dwelling.typeTenant", "len.at.res", "creditlow", "creditmedium", "house.colorred", "house.colorwhite",
                      "house.coloryellow", "ni.age", "zip.codeAZ", "zip.codeCO", "zip.codeDC", "zip.codeWA",
                      "zip.codePA", "n.children.n.adults", "ni.genderM.ni.marital.status", "coverage.typeB.dwelling.typeHouse",
                      "coverage.typeB.dwelling.typeTenant", "coverage.typeC.dwelling.typeHouse",
                      "coverage.typeC.dwelling.typeTenant", "ni.marital.status.ni.age", "ni.marital.status.coverage.typeB",
                      "ni.marital.status.coverage.typeC", "coverage.typeB.ni.age", "coverage.typeC.ni.age",
                      "sales.channelNonBroker.ni.age")
My.stepwise.glm(Y="cancel", variable.list = my.variable.list, data=train1, sle = 0.05,
                        sls = 0.05, myfamily = "binomial")

# reduced model after stepwise selection
mod1 <- glm(cancel ~ creditlow +n.children.n.adults +creditmedium +zip.codeDC +zip.codePA +ni.marital.status.ni.age
            +claim.ind +len.at.res +ni.age +tenure +sales.channelNonBroker.ni.age +zip.codeCO +coverage.typeB.ni.age
            +coverage.typeB.dwelling.typeTenant, family=binomial ,data=train1)
summary(mod1) 


# ======================================================== #
# 4. Model diagnostics & influential Observation detection #
# ======================================================== #

hii   <- hatvalues(mod1)
p.res <- residuals(mod1, type="pearson")
d.res <- residuals(mod1, type ="deviance")
delchi <- (p.res/(sqrt(1-hii)))^2
deldev <- hii*delchi+d.res^2
CooksD <- cooks.distance(mod1)

plot(hii, main="Hat Diagonals")
plot(delchi, main="Delta Chi-Squares")
plot(deldev, main="Delta Deviances")
plot(CooksD, main="Cook's Distances")
phat  <- mod1$fitted.values
plot( mod1$linear.predictors, d.res, xlab= "linear predictor", ylab="dev. resid")

train1$hii <- hii
train1$delchi <- delchi
train1$deldev <- deldev
train1$CooksD <- CooksD

train1 <- subset(train1,train1$CooksD < 0.003)
train1 <- subset(train1,train1$delchi < 30)
train1 <- subset(train1,train1$hii < 0.013)

mod0 <- glm(cancel ~ tenure +claim.ind +n.adults +n.children +ni.gender +ni.marital.status +premium
            +sales.channelNonBroker +coverage.type +dwelling.type +len.at.res +credit +house.color +ni.age
            +zip.codeAZ +zip.codeCO +zip.codeDC +zip.codeIA +zip.codePA +n.children*n.adults +ni.gender*ni.marital.status
            +coverage.type*dwelling.type +coverage.type*ni.age +sales.channelNonBroker*ni.age +ni.marital.status*ni.age
            +ni.marital.status*coverage.type, family=binomial, data=train1)
summary(mod0)

# Stepwise regression while checking VIF simultaneously
my.variable.list <- c("tenure", "claim.ind", "n.adults", "n.children", "ni.genderM", "ni.marital.status", "premium",
                      "sales.channelNonBroker", "coverage.typeB", "coverage.typeC", "dwelling.typeHouse",
                      "dwelling.typeTenant", "len.at.res", "creditlow", "creditmedium", "house.colorred", "house.colorwhite",
                      "house.coloryellow", "ni.age", "zip.codeAZ", "zip.codeCO", "zip.codeDC", "zip.codeWA",
                      "zip.codePA", "n.children.n.adults", "ni.genderM.ni.marital.status", "coverage.typeB.dwelling.typeHouse",
                      "coverage.typeB.dwelling.typeTenant", "coverage.typeC.dwelling.typeHouse",
                      "coverage.typeC.dwelling.typeTenant", "ni.marital.status.ni.age", "ni.marital.status.coverage.typeB",
                      "ni.marital.status.coverage.typeC", "coverage.typeB.ni.age", "coverage.typeC.ni.age",
                      "sales.channelNonBroker.ni.age")
My.stepwise.glm(Y="cancel", variable.list = my.variable.list, data=train1, sle = 0.05,
                sls = 0.05, myfamily = "binomial")

# reduced model after stepwise selection
mod1 <- glm(cancel ~ creditlow +n.children.n.adults +creditmedium +zip.codeDC +zip.codePA +ni.marital.status.ni.age
            +claim.ind +len.at.res +ni.age +tenure +sales.channelNonBroker.ni.age +zip.codeCO +coverage.typeB.ni.age
            +coverage.typeB.dwelling.typeTenant, family=binomial ,data=train1)
summary(mod1)

hii   <- hatvalues(mod1)
p.res <- residuals(mod1, type="pearson")
d.res <- residuals(mod1, type ="deviance")
delchi <- (p.res/(sqrt(1-hii)))^2
deldev <- hii*delchi+d.res^2
CooksD <- cooks.distance(mod1)

plot(hii, main="Hat Diagonals")
plot(delchi, main="Delta Chi-Squares")
plot(deldev, main="Delta Deviances")
plot(CooksD, main="Cook's Distances")
phat  <- mod1$fitted.values
plot( mod1$linear.predictors, d.res, xlab= "linear predictor", ylab="dev. resid")

# Extra Deviance test for H0: beta(ni.genderM)=beta(premium)=...=0
G2  <- mod1$deviance - mod0$deviance
dfs <- mod1$df.residual - mod0$df.residual
qchisq(0.05, dfs, lower.tail=FALSE)

# Hosmer-Lemeshow goodness of fit test
hl <- hoslem.test(train1$cancel, fitted(mod1), g=20)
hl


# =================== #
# 5. Model evaluation #
# =================== #

# Classification Table
# using p=.25 as threshold for classification:
# this is not based on cross-validated
# predicted probabilities as ctable option in 
# SAS Proc Logistic gives.
tab <- table( train1$cancel, mod1$fitted.values>.25)
addmargins(tab)

# ROC curve, AUC
pred <- prediction(mod1$fitted.values, train1$cancel)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
abline(0,1)
performance(pred,"auc")


# ============================================= #
# 6. Model validation (k fold cross validation) #
# ============================================= #

folds <- createFolds(train1$id, k = 10, list = TRUE, returnTrain = FALSE)
accuracy <- data.frame(fold = as.numeric(), AUC = as.numeric())

for(i in 1:10) {
  k_fold_test <- train1[folds[[i]],]
  k_fold_train <- train1[-folds[[i]],]
  
  model <- glm(cancel ~ creditlow +n.children.n.adults +creditmedium +zip.codeDC +zip.codePA +ni.marital.status.ni.age
               +claim.ind +len.at.res +ni.age +tenure +sales.channelNonBroker.ni.age +zip.codeCO +coverage.typeB.ni.age
               +coverage.typeB.dwelling.typeTenant, family=binomial, data = k_fold_train)
  x <- predict(model, newdata = k_fold_test)
  y = exp(x)/(1+exp(x))
  pred <- prediction(y, k_fold_test$cancel)
  auc <- performance(pred,"auc")
  
  accuracy[i,1] <- i
  accuracy[i,2] <- auc@y.values
}
accuracy[i+1,1] <- "mean"
accuracy[i+1,2] <- mean(accuracy[-(i+1),2])


# ========================= #
# 7. Prediction on test set #
# ========================= #

test = read.csv("Test.csv")

# Erasing data points that are presumptively wrong or new classes
test$ni.age[test$ni.age > 100] <- NA
test$len.at.res[test$ni.age - test$len.at.res < 0] <- NA
test$dwelling.type[test$dwelling.type == "Landlord"] <- NA
test$ni.gender = as.factor(test$ni.gender)
test$sales.channel = as.factor(test$sales.channel)
test$coverage.type = as.factor(test$coverage.type)
test$dwelling.type = as.factor(test$dwelling.type)
test$credit = as.factor(test$credit)

# Replace NA with predicted values that are regressed by other data points
impdata <- mice(test, m=1, maxit = 50, method = 'pmm', seed = 500)
test1 <- complete(impdata)

write.csv(test1, file = "Testimp.csv", row.names = FALSE)

# Creating dummy variables in imputed test set
test1$ni.genderM <- ifelse(test1$ni.gender == "M", 1, 0)
test1$sales.channelNonBroker <- ifelse(test1$sales.channel != "Broker", 1, 0)
test1$coverage.typeA <- ifelse(test1$coverage.type == "A", 1, 0)
test1$coverage.typeB <- ifelse(test1$coverage.type == "B", 1, 0)
test1$coverage.typeC <- ifelse(test1$coverage.type == "C", 1, 0)
test1$dwelling.typeCondo <- ifelse(test1$dwelling.type == "Condo", 1, 0)
test1$dwelling.typeHouse <- ifelse(test1$dwelling.type == "House", 1, 0)
test1$dwelling.typeTenant <- ifelse(test1$dwelling.type == "Tenant", 1, 0)
test1$creditlow <- ifelse(test1$credit == "low", 1, 0)
test1$creditmedium <- ifelse(test1$credit == "medium", 1, 0)
test1$zip.codeCO <- ifelse(test1$zip.code >= 80000 & test1$zip.code < 81000, 1, 0)
test1$zip.codeDC <- ifelse(test1$zip.code >= 20000 & test1$zip.code < 21000, 1, 0)
test1$zip.codePA <- ifelse(test1$zip.code >= 15000 & test1$zip.code < 16000, 1, 0)
test1$n.children.n.adults <- test1$n.children*test1$n.adults
test1$ni.genderM.ni.marital.status <- test1$ni.genderM*test1$ni.marital.status
test1$coverage.typeB.dwelling.typeHouse <- test1$coverage.typeB*test1$dwelling.typeHouse
test1$coverage.typeB.dwelling.typeTenant <- test1$coverage.typeB*test1$dwelling.typeTenant
test1$coverage.typeC.dwelling.typeHouse <- test1$coverage.typeC*test1$dwelling.typeHouse
test1$coverage.typeC.dwelling.typeTenant <- test1$coverage.typeC*test1$dwelling.typeTenant
test1$ni.marital.status.ni.age <- test1$ni.marital.status*test1$ni.age
test1$ni.marital.status.coverage.typeB <- test1$ni.marital.status*test1$coverage.typeB
test1$ni.marital.status.coverage.typeC <- test1$ni.marital.status*test1$coverage.typeC
test1$coverage.typeB.ni.age <- test1$coverage.typeB*test1$ni.age
test1$coverage.typeC.ni.age <- test1$coverage.typeC*test1$ni.age
test1$sales.channelNonBroker.ni.age <- test1$sales.channelNonBroker*test1$ni.age

x = predict(mod1, newdata = test1)
test1$pred = exp(x)/(1+exp(x))
test2 <- test1[,c("id", "pred")]
write.csv(test2, file = "Submission.csv", row.names = FALSE)