get_metrics_train_test <- function(t_train, t_test, l){
  train_accuracy <- sum(diag(t_train))/sum(t_train)
  train_precision <- 0
  train_recall <- 0
  train_F1 <- 0
  for(i in 1:l){
    temp_prec <- t_train[i,i]/sum(t_train[,i])
    temp_rec <- t_train[i,i]/sum(t_train[i,])
    train_precision <- train_precision + temp_prec
    train_recall <- train_recall + temp_rec
    train_F1 <- train_F1 + 2*(temp_prec*temp_rec)/(temp_rec+temp_prec)
  }
  train_precision <- train_precision/l
  train_recall <- train_recall/l
  train_F1 <- train_F1/l
  
  test_accuracy <- sum(diag(t_test))/sum(t_test)
  test_precision <- 0
  test_recall <- 0
  test_F1 <- 0
  for(i in 1:l){
    temp_prec <- t_test[i,i]/sum(t_test[,i])
    temp_rec <- t_test[i,i]/sum(t_test[i,])
    test_precision <- test_precision + temp_prec
    test_recall <- test_recall + temp_rec
    test_F1 <- test_F1 + 2*(temp_prec*temp_rec)/(temp_rec+temp_prec)
  }
  test_precision <- test_precision/l
  test_recall <- test_recall/l
  test_F1 <- test_F1/l
  
  accuracies = cbind(training = train_accuracy,
                         test = test_accuracy)
  precision = cbind(training = train_precision,
                        test = test_precision)
  recall = cbind(training = train_recall,
                     test = test_recall)
  F1 = cbind(training = train_F1,
                 test = test_F1)
  metrics = rbind(accuracy = accuracies,
                      precision = precision,
                      recall = recall,
                      F1_score = F1)
  row.names(metrics) <- c("accuracy", "precision", "recall", "F1_score")
  return(metrics)
}
get_metrics_test <- function(t_test, l){

  test_accuracy <- sum(diag(t_test))/sum(t_test)
  test_precision <- 0
  test_recall <- 0
  test_F1 <- 0
  for(i in 1:l){
    temp_prec <- t_test[i,i]/sum(t_test[,i])
    temp_rec <- t_test[i,i]/sum(t_test[i,])
    test_precision <- test_precision + temp_prec
    test_recall <- test_recall + temp_rec
    test_F1 <- test_F1 + 2*(temp_prec*temp_rec)/(temp_rec+temp_prec)
  }
  test_precision <- test_precision/l
  test_recall <- test_recall/l
  test_F1 <- test_F1/l
  
  metrics = rbind(accuracy = test_accuracy,
                  precision = test_precision,
                  recall = test_recall,
                  F1_score = test_F1)
  row.names(metrics) <- c("accuracy", "precision", "recall", "F1_score")
  return(metrics)
}