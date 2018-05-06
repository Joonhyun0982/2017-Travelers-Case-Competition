#=============================================================================
# 2017 Travelers Case Competition
#     Method: Logestic Regression
#=============================================================================


# ============== #
# 1. Preparation #
# ============== #


#install.packages("caret")
#install.packages("My.stepwise")
#install.packages("ResourceSelection")
#install.packages("ROCR")
#install.packages("mice")

library("caret")  #folds creation
library("My.stepwise")  #Stepwise selection
library("ResourceSelection")  #Hosmer-Lemeshow GOF statistic
library("ROCR")  #to construct ROC curve
library("mice")  #for missing data imputation on test set

train = read.csv("Train.csv")
head(train)

# Data cleaning
train1 <- subset(train, train$cancel!=-1)
train1 <- subset(train1, train1$ni.age < 100)
train1 <- subset(train1, train1$ni.age - train1$len.at.res > 0)
train1 <- train1[complete.cases(train1[,1:18]),] #deleting rows with NA. Can be done by train1 <- na.omit(train1) as well

# Constructing dummy variables for qualitative variables
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
train1$year2013 <- ifelse(train1$year == 2013, 1, 0)
train1$year2014 <- ifelse(train1$year == 2014, 1, 0)
train1$year2015 <- ifelse(train1$year == 2015, 1, 0)
train1$year2016 <- ifelse(train1$year == 2016, 1, 0)
train1$zip.codeAZ <- ifelse(train1$zip.code >= 85000 & train1$zip.code < 86000, 1, 0)
train1$zip.codeCO <- ifelse(train1$zip.code >= 80000 & train1$zip.code < 81000, 1, 0)
train1$zip.codeDC <- ifelse(train1$zip.code >= 20000 & train1$zip.code < 21000, 1, 0)
train1$zip.codeIA <- ifelse(train1$zip.code >= 50000 & train1$zip.code < 51000, 1, 0)
train1$zip.codePA <- ifelse(train1$zip.code >= 15000 & train1$zip.code < 16000, 1, 0)
train1$zip.codeWA <- ifelse(train1$zip.code >= 98000 & train1$zip.code < 99000, 1, 0)

# Removing redundant variables
train1 <- subset(train1, select = -c(ni.gender, sales.channel, coverage.type, dwelling.type, credit, house.color, year, zip.code))


# ============================ #
# 2. Exploratory data analysis #
# ============================ #

# Correlation matrix
train1.cov <- cor(train1[,2:37])

mod0 <- glm(cancel ~ tenure +claim.ind +n.adults +n.children +ni.genderM +ni.marital.status +premium
            +sales.channelOnline +sales.channelPhone +coverage.typeB +coverage.typeC +dwelling.typeHouse
            +dwelling.typeTenant +len.at.res +creditlow +creditmedium +house.colorred +house.colorwhite
            +house.coloryellow +year2014 +year2015 +year2016 +ni.age +train1$zip.codeAZ
            +train1$zip.codeCO +train1$zip.codeDC +train1$zip.codeIA +train1$zip.codePA, family=binomial, data=train1)  # full model
summary(mod0)

# Check to see if there is a multicollinearity problem
vif <- vif(mod0)
vif

# Tests for regrouping categorical variables
tst <- glm(cancel ~ sales.channelOnline +sales.channelPhone, family=binomial, data = train1)
summary(tst)
tst <- glm(cancel ~ sales.channelBroker +sales.channelPhone, family=binomial, data = train1)
summary(tst)
tst <- glm(cancel ~ sales.channelBroker +sales.channelOnline, family=binomial, data = train1)
summary(tst)  # We see sales.channelOnline and sales.channelPhone can be grouped
# Repeat these tests for all variables with 3 or more categories


# ========================== #
# 3. Model fitting/selection #
# ========================== #

# Reconstructing dummy variables after initial examination
train1$sales.channelNonBroker <- ifelse(train1$sales.channelBroker == 0, 1, 0)
train1$year13and16 <- ifelse(train1$year2013 == 1 | train1$year2016 ==1, 1, 0)

# Stepwise regression
my.variable.list <- c("tenure", "claim.ind", "n.adults", "n.children", "ni.genderM", "ni.marital.status", "premium",
                      "sales.channelNonBroker", "coverage.typeB", "coverage.typeC", "dwelling.typeHouse",
                      "dwelling.typeTenant", "len.at.res", "creditlow", "creditmedium", "house.colorred", "house.colorwhite",
                      "house.coloryellow", "year2014", "year2015", "ni.age", "zip.codeAZ",
                      "zip.codeCO", "zip.codeDC", "zip.codeWA", "zip.codePA")
My.stepwise.glm(Y="cancel", variable.list = my.variable.list, data=train1, sle = 0.05,
                        sls = 0.05, myfamily = "binomial")

# reduced model after stepwise selection
mod1 <- glm(cancel ~ creditlow +sales.channelNonBroker +creditmedium +zip.codeDC +n.children +year2014 
            +zip.codePA +claim.ind +year2015 +ni.age +len.at.res +ni.marital.status +tenure +n.adults
            +zip.codeCO, family=binomial ,data=train1)
summary(mod1)


# ======================================================== #
# 4. Model diagnostics & influencial Observation detection #
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

mod0 <- glm(cancel ~ tenure +claim.ind +n.adults +n.children +ni.genderM +ni.marital.status +premium
            +sales.channelNonBroker +coverage.typeB +coverage.typeC +dwelling.typeHouse
            +dwelling.typeTenant +len.at.res +creditlow +creditmedium +house.colorred +house.colorwhite
            +house.coloryellow +year2014 +year2015 +ni.age +train1$zip.codeAZ
            +train1$zip.codeCO +train1$zip.codeDC +train1$zip.codeWA +train1$zip.codePA, family=binomial, data=train1)

# Stepwise regression
my.variable.list <- c("tenure", "claim.ind", "n.adults", "n.children", "ni.genderM", "ni.marital.status", "premium",
                      "sales.channelNonBroker", "coverage.typeB", "coverage.typeC", "dwelling.typeHouse",
                      "dwelling.typeTenant", "len.at.res", "creditlow", "creditmedium", "house.colorred", "house.colorwhite",
                      "house.coloryellow", "year2014", "year2015", "ni.age", "zip.codeAZ",
                      "zip.codeCO", "zip.codeDC", "zip.codeWA", "zip.codePA")
My.stepwise.glm(Y="cancel", variable.list = my.variable.list, data=train1, sle = 0.05,
                sls = 0.05, myfamily = "binomial")

# reduced model without outliers
mod1 <- glm(cancel ~ creditlow +sales.channelNonBroker +creditmedium +zip.codeDC +n.children +year2014 
            +zip.codePA +claim.ind +year2015 +ni.age +len.at.res +ni.marital.status +tenure +n.adults
            +zip.codeCO, family=binomial ,data=train1)
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
hl <- hoslem.test(train1$cancel, fitted(mod1), g=10)
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
pred <- prediction(mod1$fitted.values,train1$cancel)
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
  
  model <- glm(cancel ~ creditlow +sales.channelNonBroker +creditmedium +zip.codeDC +n.children +year2014 
               +zip.codePA +claim.ind +year2015 +ni.age +len.at.res +ni.marital.status +tenure +n.adults
               +zip.codeCO, family=binomial, data = k_fold_train)
  x <- predict(model, newdata = k_fold_test)
  y = exp(x)/(1+exp(x))
  pred <- prediction(y, k_fold_test$cancel)
  acc <- performance(pred,"auc")
  
  accuracy[i,1] <- i
  accuracy[i,2] <- acc@y.values
}
accuracy[i+1,1] <- "mean"
accuracy[i+1,2] <- mean(accuracy[-(i+1),2])


# ========================= #
# 7. Prediction on test set #
# ========================= #

test = read.csv("Test.csv")

# Erasing data points that are presumptively wrong
test$ni.age[test$ni.age > 100] <- NA
test$len.at.res[test$ni.age - test$len.at.res < 0] <- NA
test$ni.gender = as.factor(test$ni.gender)
test$sales.channel = as.factor(test$sales.channel)
test$coverage.type = as.factor(test$coverage.type)
test$dwelling.type = as.factor(test$dwelling.type)
test$credit = as.factor(test$credit)

# Replace NA with predicted values that are regressed by other data points
impdata <- mice(test, m=1, maxit = 50, method = 'pmm', seed = 500)
test1 <- complete(impdata)

# write.csv(test1, file = "Testimp.csv", row.names = FALSE)

test1$sales.channelNonBroker <- ifelse(test1$sales.channel != "Broker", 1, 0)
test1$creditlow <- ifelse(test1$credit == "low", 1, 0)
test1$creditmedium <- ifelse(test1$credit == "medium", 1, 0)
test1$year2014 <- ifelse(test1$year == 2014, 1, 0)
test1$year2015 <- ifelse(test1$year == 2015, 1, 0)
test1$zip.codeCO <- ifelse(test1$zip.code >= 80000 & test1$zip.code < 81000, 1, 0)
test1$zip.codeDC <- ifelse(test1$zip.code >= 20000 & test1$zip.code < 21000, 1, 0)
test1$zip.codePA <- ifelse(test1$zip.code >= 15000 & test1$zip.code < 16000, 1, 0)

x = predict.glm(mod1, newdata = test1)
test1$pred = exp(x)/(1+exp(x))
test1$cancel = ifelse(test1$pred > 0.5, 1, 0)
test2 <- test1[,c("id", "cancel")]
write.csv(test2, file = "Submission.csv", row.names = FALSE)