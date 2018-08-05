### India Hacks challenge | Modelling - xgboost ###
### Author: Naveen Kaveti | Date: 17-01-2016 ###

### Setting working directory ###
filepath <- c("/Users/nkaveti/Documents/Kaggle/will_bill_solve_it")
setwd(filepath)

### Loading Required packages ###
library(caret)
library(randomForest)
library(Matrix)
library(e1071)
library(nnet)

### Reading data into R ###
train <- read.csv("train/fulldata_train_v2.csv")
test <- read.csv("test/fulldata_test_v2.csv")

train_pred <- read.csv("solution/train_pred_rf_nnet_logit.csv")
test_pred <- read.csv("solution/test_pred_rf_nnet_logit.csv")

user_accuracy <- (train$solved_count.y) / (train$solved_count.y + train$attempts)

train <- train[,c("user_id", "problem_id", "target", "level", "accuracy", "user_type", "prob_accuracy")]
train <- cbind(train, user_accuracy = user_accuracy)
train$level <- factor(train$level, ordered = TRUE)
train$user_type <- factor(train$user_type)

user_accuracy_te <- (test$solved_count.y) / (test$solved_count.y + test$attempts)
test <- test[,c("user_id", "problem_id", "Id", "level", "accuracy", "user_type", "prob_accuracy")]
test <- cbind(test, user_accuracy = user_accuracy_te)
test$level <- factor(test$level, ordered = TRUE)
test$user_type <- factor(test$user_type)

### Random Forest Model ###
num_folds <- 2
num_tree <- 500
folds <- createFolds(as.factor(train$target), k = num_folds, list = FALSE)

train_pred_rf <- c()
test_pred_rf <- data.frame(matrix(0,nrow = nrow(test), ncol = num_folds + 1))
colnames(test_pred_rf) <- c("Id", "solved_status_1", "solved_status_2")
test_pred_rf$Id <- test$Id

for(i in 1:num_folds){
  rf_model <- randomForest(as.factor(train[folds != i,"target"]) ~., data = train[!(folds == i), -c(1:3)], ntree = num_tree, norm.votes = TRUE)
  pre <- predict(rf_model, train[folds == i, -c(1:3)], type = "prob")[,"1"]
  tab <- table(pre, train[folds == i, "target"])
  acc <- sum(diag(tab))/sum(tab)
  cat("Accuracy of fold ", i, " is ", acc, "\n")
  train_pred_rf <- rbind(train_pred_rf, cbind(train[folds == i, c(1:3)], predictions = pre))
  test_pred_rf[,i+1] <- predict(rf_model, test[,-c(1:3)], type = "prob")[,"1"]
  cat("Completed fold ", i, "\n")
}


### Logistic ###

train_pred_logit <- c()
test_pred_logit <- data.frame(matrix(0,nrow = nrow(test), ncol = num_folds + 1))
colnames(test_pred_logit) <- c("Id", "solved_status_1", "solved_status_2")
test_pred_logit$Id <- test$Id

for(i in 1:num_folds){
  logit_model <- glm(target ~., data = train[folds != i, -c(1:2)], family = binomial("logit"), maxit = 100)
  #cat(summary(logit_model), "\n")
  pre <- predict(logit_model, train[folds == i, -c(1:3)], type = "response")
  tab <- table(pre, train[folds == i, "target"])
  acc <- sum(diag(tab))/sum(tab)
  cat("Accuracy of fold ", i, " is ", acc, "\n")
  train_pred_logit <- rbind(train_pred_logit, cbind(train[folds == i, c(1:3)], predictions = pre))
  test_pred_logit[,i+1] <- predict(logit_model, test[,-c(1,3)], type = "response")
  cat("Completed fold ", i, "\n")
}

### Neural Netwrok ###
set.seed(3424)
train_pred_nnet <- c()
test_pred_nnet <- data.frame(matrix(0,nrow = nrow(test), ncol = num_folds + 1))
colnames(test_pred_nnet) <- c("Id", "solved_status_1", "solved_status_2")
test_pred_nnet$Id <- test$Id

for(i in 1:num_folds){
  nnet_model <- nnet(as.factor(target) ~ ., data = train[folds != i, -c(1:2)], maxit = 500, size = 5)
  #cat(summary(logit_model), "\n")
  pre <- predict(nnet_model, train[folds == i, -c(1:3)], type = "raw")
  tab <- table(pre, train[folds == i, "target"])
  acc <- sum(diag(tab))/sum(tab)
  cat("Accuracy of fold ", i, " is ", acc, "\n")
  train_pred_nnet <- rbind(train_pred_nnet, cbind(train[folds == i, c(1:3)], predictions = pre))
  test_pred_nnet[,i+1] <- predict(nnet_model, test[,-c(1,3)], type = "raw")
  cat("Completed fold ", i, "\n")
}

### Merging all the solutions ###
train_pred <- train_pred_logit
colnames(train_pred)[ncol(train_pred)] <- "predictions_logit"
train_pred$predictions_nnet <- train_pred_nnet$predictions
train_pred$predictions_rf <- train_pred_rf$predictions

write.csv(train_pred, file = "solution/train_pred_rf_nnet_logit.csv")

test_pred <- test_pred_logit
colnames(test_pred)[2:3] <- c("solved_status_1_logit", "solved_status_2_logit")
test_pred$solved_status_1_nnet <- test_pred_nnet$solved_status_1
test_pred$solved_status_2_nnet <- test_pred_nnet$solved_status_2
test_pred$solved_status_1_rf <- test_pred_rf$solved_status_1
test_pred$solved_status_2_rf <- test_pred_rf$solved_status_2

write.csv(test_pred, file = "solution/test_pred_rf_nnet_logit.csv")


sapply(c(4:6), FUN = function(x,data,cutoff){
  tr_pre <- ifelse(data[,x] < cutoff, 0, 1)
  tab <- table(tr_pre, data$target)
  cat("Accuracy := ", sum(diag(tab))/sum(tab), "\n")
}, data = train_pred, cutoff = 0.45)

train_pred$simple_avg <- (train_pred$predictions_logit + train_pred$predictions_nnet + train_pred$predictions_nnet)/3

tab <- table(ifelse(train_pred$simple_avg < 0.45, 0, 1), train_pred$target)
acc <- sum(diag(tab))/sum(tab)

### Logistic Regression on predictions ###
stacking <- glm(target ~., data = train_pred[folds != 1,-c(1,2,7)], family = binomial("logit"), maxit = 100)

stacking_rf <- randomForest(as.factor(target) ~., data = train_pred[folds != 1,-c(1,2,7)], ntree = 500, norm.votes = TRUE)

pred <- predict(stacking, train_pred[folds == 1, -c(1,2,7)], type = "response")
pred_rf <- predict(stacking_rf, train_pred[folds == 1, -c(1,2,7)], type = "prob")

pred2 <- ifelse(pred < 0.5, 0, 1)
pred2_rf <- ifelse(pred_rf[,"1"] < 0.5, 0, 1)



