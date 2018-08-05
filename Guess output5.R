### India Hacks challenge | Modelling - xgboost ###
### Author: Naveen Kaveti | Date: 17-01-2016 ###

### Setting working directory ###
filepath <- c("/Users/nkaveti/Documents/Kaggle/will_bill_solve_it")
setwd(filepath)

### Loading Required packages ###
library(caret)
library(xgboost)
library(Matrix)

### Reading data into R ###
train <- read.csv("train/fulldata_train_v2.csv")
test <- read.csv("test/fulldata_test_v2.csv")
rf_result_tr <- read.csv("solution/Old solutions/train_pred_rf_prob.csv")
rf_result_te <- read.csv("solution/Old solutions/test_pred_rf_prob.csv")

# test <- test[,-which(colnames(test) %in% colnames(train)[13:90])]
# train <- train[,-c(13:90)]

user_accuracy <- (train$solved_count.y) / (train$solved_count.y + train$attempts)

train <- train[,c("user_id", "problem_id", "target", "level", "accuracy", "user_type", "prob_accuracy")]
train <- cbind(train, user_accuracy = user_accuracy)
train$level <- factor(train$level, ordered = TRUE)
train$user_type <- factor(train$user_type)

train$dummy <- paste0(train$user_id,"_",train$problem_id)
train_pred$dummy <- paste0(train_pred$user_id, "_", train_pred$problem_id)

train <- merge(train, train_pred[c("dummy", "predictions")], by = "dummy")
train$dummy <- NULL

user_accuracy_te <- (test$solved_count.y) / (test$solved_count.y + test$attempts)
test <- test[,c("user_id", "problem_id", "Id", "level", "accuracy", "user_type", "prob_accuracy")]
test <- cbind(test, user_accuracy = user_accuracy_te)
test$level <- factor(test$level, ordered = TRUE)
test$user_type <- factor(test$user_type)

test_pred$predictions <- (test_pred$solved_status_1 + test_pred$solved_status_2)/2

test <- merge(test, test_pred[c("Id", "predictions")], by = "Id")


train$level <- factor(train$level, ordered = TRUE)
test$level <- factor(test$level, ordered = TRUE)
train$user_type <- factor(train$user_type)
test$user_type <- factor(test$user_type)
test <- test[order(test$Id),] # Sorting test data based on ID column

# train$problem_id <- factor(train$problem_id, levels = unique(c(train$problem_id,test$problem_id)))
# test$problem_id <- factor(test$problem_id, levels = unique(c(train$problem_id,test$problem_id)))

### Feature importance ###
num_folds <- 2
# num_tree <- 200
folds <- createFolds(as.factor(train$target), k = num_folds, list = FALSE)
train_pred <- c()
test_pred <- data.frame(matrix(0,nrow = nrow(test), ncol = num_folds + 1))
colnames(test_pred) <- c("Id", "solved_status_1", "solved_status_2")
test_pred$Id <- test$Id

params <- list(booster = "gbtree", eta = 0.05, gamma = 0.0, max_depth = 6, min_child_weight = 10, subsample = 0.6, colsample_bytree = 0.8, objective = "binary:logistic")

for(i in 1:num_folds){
  train_sparse <- sparse.model.matrix(~., data = train[!(folds == i), -c(1:3)])
  test_sparse <- sparse.model.matrix(~., data = train[folds == i, -c(1:3)])
  train_xgb <- xgb.DMatrix(train_sparse, label = train[!(folds == i), "target"])
  test_xgb <- xgb.DMatrix(test_sparse, label = train[folds == i, "target"])
  watchlist <- list(eval = test_xgb, train = train_xgb)
  xgb_model <- xgb.train(params = params, train_xgb, eval_metric = "error", nrounds = 500, watchlist = watchlist, early.stop.round = 30)
  pre <- predict(xgb_model, test_xgb)
  train_pred <- rbind(train_pred, cbind(train[folds == i, c(1:3)], predictions = pre))
  te_pre <- predict(xgb_model, sparse.model.matrix(~., data = test[,-c(1:3)]))
  #te_pre <- (te_pre - min(te_pre))/(max(te_pre) - min(te_pre))
  test_pred[,i+1] <- te_pre
  cat("Completed fold ", i, "\n")
}

write.csv(train_pred, file = "sol_xgb_train_prob_new_feat_3.csv", row.names = FALSE)
write.csv(test_pred, file = "sol_xgb_test_prob_new_feat_3.csv", row.names = FALSE)

sol <- data.frame(Id = test_pred$Id, solved_status = rowMeans(test_pred[,c(2:4)]))
sol$solved_status <- ifelse(sol$solved_status < 0.5, 0, 1)

write.csv(sol, file = "nohopes.csv", row.names = FALSE)




te_pr <- sapply(train_pred[,-1], rank)
summary((te_pr[,1] - min(te_pr[,1]))/(max(te_pr[,1]) - min(te_pr[,1])))

sol_xgb <- data.frame(Id = test$Id, solved_status = (test_pred$solved_status_1 + test_pred$solved_status_2)/2)

sol_xgb$solved_status <- (sol_xgb$solved_status - min(sol_xgb$solved_status))/(max(sol_xgb$solved_status) - min(sol_xgb$solved_status))

sol_xgb$solved_status <- (sol_xgb$solved_status - min(sol_xgb$solved_status))/(max(sol_xgb$solved_status) - min(sol_xgb$solved_status))

sol_xgb$solved_status <- ifelse(sol_xgb$solved_status <= 0.7, 0, 1)
sol_xgb <- data.frame(Id = test$Id, solved_status = te_pr)
write.csv(sol_xgb, file = "solution/sol_xgb_bad_exp2.csv", row.names = FALSE)
