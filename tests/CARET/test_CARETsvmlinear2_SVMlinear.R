library(caret)
library(foreign)
library(patrick)
library(R.utils)
library(testthat)

if(!require("e1071", character.only = TRUE ))
  install.packages("e1071", dependencies = TRUE)
library("e1071", character.only = TRUE)

paramGrid <- c(
              "cost: 1.0")

withTimeout(timeout=21600, onTimeout="error", expr={
    with_parameters_test_that("UniformSplit", test_name=paramGrid, test_params=paramGrid, code={
        for (iter in 1:1){
            set.seed(42)
            testdata <- read.arff(paste0("smokedata/UniformSplit_",iter,"_test.arff"))
            traindata <- read.arff(paste0("smokedata/UniformSplit_",iter,"_training.arff"))
            target_index <- ncol(traindata)
            train_x <- traindata[,-target_index]
            train_y <- traindata[,target_index]
            test_x  <- testdata[,-target_index]
            test_y  <- testdata[,target_index]

            paramSplit <- strsplit(test_params, ": ")
            paramNames <- unlist(strsplit(paramSplit[[1]][1], ","))
            paramSet <- read.table(text=paramSplit[[1]][2], sep=",")
            colnames(paramSet) <- paramNames

            control <- trainControl(method = "none")
            model <- caret::train(x = train_x,
                           y = train_y,
                           method = "svmLinear2",
                           tuneGrid = paramSet,
                           trControl = control
                           )
            predictions <- predict(model, test_x)
            probabilities <- array(-1,dim = c(length(test_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, test_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities <- predict(model, test_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes <- as.integer(test_y) - 1
            pred_classes <- as.integer(predictions) - 1
            csv_df <- cbind(actual = actual_classes,
                          prediction = pred_classes,
                          prob_0 = probabilities[,1],
                          prob_1 = probabilities[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_UniformSplit_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))

            # training data as test data
            predictions_training_as_test <- predict(model, train_x)
            probabilities_training_as_test <- array(-1,dim = c(length(train_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, train_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities_training_as_test <- predict(model, train_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes_training_as_test <- as.integer(train_y) - 1
            pred_classes_training_as_test <- as.integer(predictions_training_as_test) - 1
            csv_df <- cbind(actual = actual_classes_training_as_test,
                          prediction = pred_classes_training_as_test,
                          prob_0 = probabilities_training_as_test[,1],
                          prob_1 = probabilities_training_as_test[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_UniformSplit_TrainingAsTest_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))
        }
    expect_true(TRUE)
    })
})

withTimeout(timeout=21600, onTimeout="error", expr={
    with_parameters_test_that("RandomNumericSplit", test_name=paramGrid, test_params=paramGrid, code={
        for (iter in 1:1){
            set.seed(42)
            testdata <- read.arff(paste0("smokedata/RandomNumericSplit_",iter,"_test.arff"))
            traindata <- read.arff(paste0("smokedata/RandomNumericSplit_",iter,"_training.arff"))
            target_index <- ncol(traindata)
            train_x <- traindata[,-target_index]
            train_y <- traindata[,target_index]
            test_x  <- testdata[,-target_index]
            test_y  <- testdata[,target_index]

            paramSplit <- strsplit(test_params, ": ")
            paramNames <- unlist(strsplit(paramSplit[[1]][1], ","))
            paramSet <- read.table(text=paramSplit[[1]][2], sep=",")
            colnames(paramSet) <- paramNames

            control <- trainControl(method = "none")
            model <- caret::train(x = train_x,
                           y = train_y,
                           method = "svmLinear2",
                           tuneGrid = paramSet,
                           trControl = control
                           )
            predictions <- predict(model, test_x)
            probabilities <- array(-1,dim = c(length(test_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, test_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities <- predict(model, test_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes <- as.integer(test_y) - 1
            pred_classes <- as.integer(predictions) - 1
            csv_df <- cbind(actual = actual_classes,
                          prediction = pred_classes,
                          prob_0 = probabilities[,1],
                          prob_1 = probabilities[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_RandomNumericSplit_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))

            # training data as test data
            predictions_training_as_test <- predict(model, train_x)
            probabilities_training_as_test <- array(-1,dim = c(length(train_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, train_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities_training_as_test <- predict(model, train_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes_training_as_test <- as.integer(train_y) - 1
            pred_classes_training_as_test <- as.integer(predictions_training_as_test) - 1
            csv_df <- cbind(actual = actual_classes_training_as_test,
                          prediction = pred_classes_training_as_test,
                          prob_0 = probabilities_training_as_test[,1],
                          prob_1 = probabilities_training_as_test[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_RandomNumericSplit_TrainingAsTest_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))
        }
    expect_true(TRUE)
    })
})

withTimeout(timeout=21600, onTimeout="error", expr={
    with_parameters_test_that("BreastCancer", test_name=paramGrid, test_params=paramGrid, code={
        for (iter in 1:1){
            set.seed(42)
            testdata <- read.arff(paste0("smokedata/BreastCancer_",iter,"_test.arff"))
            traindata <- read.arff(paste0("smokedata/BreastCancer_",iter,"_training.arff"))
            target_index <- ncol(traindata)
            train_x <- traindata[,-target_index]
            train_y <- traindata[,target_index]
            test_x  <- testdata[,-target_index]
            test_y  <- testdata[,target_index]

            paramSplit <- strsplit(test_params, ": ")
            paramNames <- unlist(strsplit(paramSplit[[1]][1], ","))
            paramSet <- read.table(text=paramSplit[[1]][2], sep=",")
            colnames(paramSet) <- paramNames

            control <- trainControl(method = "none")
            model <- caret::train(x = train_x,
                           y = train_y,
                           method = "svmLinear2",
                           tuneGrid = paramSet,
                           trControl = control
                           )
            predictions <- predict(model, test_x)
            probabilities <- array(-1,dim = c(length(test_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, test_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities <- predict(model, test_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes <- as.integer(test_y) - 1
            pred_classes <- as.integer(predictions) - 1
            csv_df <- cbind(actual = actual_classes,
                          prediction = pred_classes,
                          prob_0 = probabilities[,1],
                          prob_1 = probabilities[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_BreastCancer_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))

            # training data as test data
            predictions_training_as_test <- predict(model, train_x)
            probabilities_training_as_test <- array(-1,dim = c(length(train_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, train_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities_training_as_test <- predict(model, train_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes_training_as_test <- as.integer(train_y) - 1
            pred_classes_training_as_test <- as.integer(predictions_training_as_test) - 1
            csv_df <- cbind(actual = actual_classes_training_as_test,
                          prediction = pred_classes_training_as_test,
                          prob_0 = probabilities_training_as_test[,1],
                          prob_1 = probabilities_training_as_test[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_BreastCancer_TrainingAsTest_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))
        }
    expect_true(TRUE)
    })
})

withTimeout(timeout=21600, onTimeout="error", expr={
    with_parameters_test_that("BreastCancerZNorm", test_name=paramGrid, test_params=paramGrid, code={
        for (iter in 1:1){
            set.seed(42)
            testdata <- read.arff(paste0("smokedata/BreastCancerZNorm_",iter,"_test.arff"))
            traindata <- read.arff(paste0("smokedata/BreastCancerZNorm_",iter,"_training.arff"))
            target_index <- ncol(traindata)
            train_x <- traindata[,-target_index]
            train_y <- traindata[,target_index]
            test_x  <- testdata[,-target_index]
            test_y  <- testdata[,target_index]

            paramSplit <- strsplit(test_params, ": ")
            paramNames <- unlist(strsplit(paramSplit[[1]][1], ","))
            paramSet <- read.table(text=paramSplit[[1]][2], sep=",")
            colnames(paramSet) <- paramNames

            control <- trainControl(method = "none")
            model <- caret::train(x = train_x,
                           y = train_y,
                           method = "svmLinear2",
                           tuneGrid = paramSet,
                           trControl = control
                           )
            predictions <- predict(model, test_x)
            probabilities <- array(-1,dim = c(length(test_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, test_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities <- predict(model, test_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes <- as.integer(test_y) - 1
            pred_classes <- as.integer(predictions) - 1
            csv_df <- cbind(actual = actual_classes,
                          prediction = pred_classes,
                          prob_0 = probabilities[,1],
                          prob_1 = probabilities[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_BreastCancerZNorm_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))

            # training data as test data
            predictions_training_as_test <- predict(model, train_x)
            probabilities_training_as_test <- array(-1,dim = c(length(train_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, train_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities_training_as_test <- predict(model, train_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes_training_as_test <- as.integer(train_y) - 1
            pred_classes_training_as_test <- as.integer(predictions_training_as_test) - 1
            csv_df <- cbind(actual = actual_classes_training_as_test,
                          prediction = pred_classes_training_as_test,
                          prob_0 = probabilities_training_as_test[,1],
                          prob_1 = probabilities_training_as_test[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_BreastCancerZNorm_TrainingAsTest_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))
        }
    expect_true(TRUE)
    })
})

withTimeout(timeout=21600, onTimeout="error", expr={
    with_parameters_test_that("BreastCancerMinMaxNorm", test_name=paramGrid, test_params=paramGrid, code={
        for (iter in 1:1){
            set.seed(42)
            testdata <- read.arff(paste0("smokedata/BreastCancerMinMaxNorm_",iter,"_test.arff"))
            traindata <- read.arff(paste0("smokedata/BreastCancerMinMaxNorm_",iter,"_training.arff"))
            target_index <- ncol(traindata)
            train_x <- traindata[,-target_index]
            train_y <- traindata[,target_index]
            test_x  <- testdata[,-target_index]
            test_y  <- testdata[,target_index]

            paramSplit <- strsplit(test_params, ": ")
            paramNames <- unlist(strsplit(paramSplit[[1]][1], ","))
            paramSet <- read.table(text=paramSplit[[1]][2], sep=",")
            colnames(paramSet) <- paramNames

            control <- trainControl(method = "none")
            model <- caret::train(x = train_x,
                           y = train_y,
                           method = "svmLinear2",
                           tuneGrid = paramSet,
                           trControl = control
                           )
            predictions <- predict(model, test_x)
            probabilities <- array(-1,dim = c(length(test_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, test_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities <- predict(model, test_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes <- as.integer(test_y) - 1
            pred_classes <- as.integer(predictions) - 1
            csv_df <- cbind(actual = actual_classes,
                          prediction = pred_classes,
                          prob_0 = probabilities[,1],
                          prob_1 = probabilities[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_BreastCancerMinMaxNorm_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))

            # training data as test data
            predictions_training_as_test <- predict(model, train_x)
            probabilities_training_as_test <- array(-1,dim = c(length(train_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, train_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities_training_as_test <- predict(model, train_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes_training_as_test <- as.integer(train_y) - 1
            pred_classes_training_as_test <- as.integer(predictions_training_as_test) - 1
            csv_df <- cbind(actual = actual_classes_training_as_test,
                          prediction = pred_classes_training_as_test,
                          prob_0 = probabilities_training_as_test[,1],
                          prob_1 = probabilities_training_as_test[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_BreastCancerMinMaxNorm_TrainingAsTest_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))
        }
    expect_true(TRUE)
    })
})

withTimeout(timeout=21600, onTimeout="error", expr={
    with_parameters_test_that("Wine", test_name=paramGrid, test_params=paramGrid, code={
        for (iter in 1:1){
            set.seed(42)
            testdata <- read.arff(paste0("smokedata/Wine_",iter,"_test.arff"))
            traindata <- read.arff(paste0("smokedata/Wine_",iter,"_training.arff"))
            target_index <- ncol(traindata)
            train_x <- traindata[,-target_index]
            train_y <- traindata[,target_index]
            test_x  <- testdata[,-target_index]
            test_y  <- testdata[,target_index]

            paramSplit <- strsplit(test_params, ": ")
            paramNames <- unlist(strsplit(paramSplit[[1]][1], ","))
            paramSet <- read.table(text=paramSplit[[1]][2], sep=",")
            colnames(paramSet) <- paramNames

            control <- trainControl(method = "none")
            model <- caret::train(x = train_x,
                           y = train_y,
                           method = "svmLinear2",
                           tuneGrid = paramSet,
                           trControl = control
                           )
            predictions <- predict(model, test_x)
            probabilities <- array(-1,dim = c(length(test_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, test_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities <- predict(model, test_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes <- as.integer(test_y) - 1
            pred_classes <- as.integer(predictions) - 1
            csv_df <- cbind(actual = actual_classes,
                          prediction = pred_classes,
                          prob_0 = probabilities[,1],
                          prob_1 = probabilities[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_Wine_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))

            # training data as test data
            predictions_training_as_test <- predict(model, train_x)
            probabilities_training_as_test <- array(-1,dim = c(length(train_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, train_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities_training_as_test <- predict(model, train_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes_training_as_test <- as.integer(train_y) - 1
            pred_classes_training_as_test <- as.integer(predictions_training_as_test) - 1
            csv_df <- cbind(actual = actual_classes_training_as_test,
                          prediction = pred_classes_training_as_test,
                          prob_0 = probabilities_training_as_test[,1],
                          prob_1 = probabilities_training_as_test[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_Wine_TrainingAsTest_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))
        }
    expect_true(TRUE)
    })
})

withTimeout(timeout=21600, onTimeout="error", expr={
    with_parameters_test_that("WineZNorm", test_name=paramGrid, test_params=paramGrid, code={
        for (iter in 1:1){
            set.seed(42)
            testdata <- read.arff(paste0("smokedata/WineZNorm_",iter,"_test.arff"))
            traindata <- read.arff(paste0("smokedata/WineZNorm_",iter,"_training.arff"))
            target_index <- ncol(traindata)
            train_x <- traindata[,-target_index]
            train_y <- traindata[,target_index]
            test_x  <- testdata[,-target_index]
            test_y  <- testdata[,target_index]

            paramSplit <- strsplit(test_params, ": ")
            paramNames <- unlist(strsplit(paramSplit[[1]][1], ","))
            paramSet <- read.table(text=paramSplit[[1]][2], sep=",")
            colnames(paramSet) <- paramNames

            control <- trainControl(method = "none")
            model <- caret::train(x = train_x,
                           y = train_y,
                           method = "svmLinear2",
                           tuneGrid = paramSet,
                           trControl = control
                           )
            predictions <- predict(model, test_x)
            probabilities <- array(-1,dim = c(length(test_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, test_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities <- predict(model, test_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes <- as.integer(test_y) - 1
            pred_classes <- as.integer(predictions) - 1
            csv_df <- cbind(actual = actual_classes,
                          prediction = pred_classes,
                          prob_0 = probabilities[,1],
                          prob_1 = probabilities[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_WineZNorm_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))

            # training data as test data
            predictions_training_as_test <- predict(model, train_x)
            probabilities_training_as_test <- array(-1,dim = c(length(train_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, train_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities_training_as_test <- predict(model, train_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes_training_as_test <- as.integer(train_y) - 1
            pred_classes_training_as_test <- as.integer(predictions_training_as_test) - 1
            csv_df <- cbind(actual = actual_classes_training_as_test,
                          prediction = pred_classes_training_as_test,
                          prob_0 = probabilities_training_as_test[,1],
                          prob_1 = probabilities_training_as_test[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_WineZNorm_TrainingAsTest_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))
        }
    expect_true(TRUE)
    })
})

withTimeout(timeout=21600, onTimeout="error", expr={
    with_parameters_test_that("WineMinMaxNorm", test_name=paramGrid, test_params=paramGrid, code={
        for (iter in 1:1){
            set.seed(42)
            testdata <- read.arff(paste0("smokedata/WineMinMaxNorm_",iter,"_test.arff"))
            traindata <- read.arff(paste0("smokedata/WineMinMaxNorm_",iter,"_training.arff"))
            target_index <- ncol(traindata)
            train_x <- traindata[,-target_index]
            train_y <- traindata[,target_index]
            test_x  <- testdata[,-target_index]
            test_y  <- testdata[,target_index]

            paramSplit <- strsplit(test_params, ": ")
            paramNames <- unlist(strsplit(paramSplit[[1]][1], ","))
            paramSet <- read.table(text=paramSplit[[1]][2], sep=",")
            colnames(paramSet) <- paramNames

            control <- trainControl(method = "none")
            model <- caret::train(x = train_x,
                           y = train_y,
                           method = "svmLinear2",
                           tuneGrid = paramSet,
                           trControl = control
                           )
            predictions <- predict(model, test_x)
            probabilities <- array(-1,dim = c(length(test_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, test_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities <- predict(model, test_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes <- as.integer(test_y) - 1
            pred_classes <- as.integer(predictions) - 1
            csv_df <- cbind(actual = actual_classes,
                          prediction = pred_classes,
                          prob_0 = probabilities[,1],
                          prob_1 = probabilities[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_WineMinMaxNorm_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))

            # training data as test data
            predictions_training_as_test <- predict(model, train_x)
            probabilities_training_as_test <- array(-1,dim = c(length(train_y),2))
            tryCatch(
                expr = {
                    if (any(is.na(predict(model, train_x, type = "prob")))){
                        stop("At least one of the predicted probability values is NaN.")
                    }
                    probabilities_training_as_test <- predict(model, train_x, type = "prob")
                },
                error = function(e){
                    message("The prediction of the probabilities failed. Values set to default (-1).")
                }
            )
            actual_classes_training_as_test <- as.integer(train_y) - 1
            pred_classes_training_as_test <- as.integer(predictions_training_as_test) - 1
            csv_df <- cbind(actual = actual_classes_training_as_test,
                          prediction = pred_classes_training_as_test,
                          prob_0 = probabilities_training_as_test[,1],
                          prob_1 = probabilities_training_as_test[,2])
            csv_file <- file.path(dirname(dirname(getwd())),"predictions",paste0("pred_CARETsvmlinear2_SVMlinear_WineMinMaxNorm_TrainingAsTest_",iter,".csv"))
            write.csv(x = csv_df,
                    file = csv_file,
                    row.names = FALSE)
            print(paste0("Predictions saved at: ", csv_file))
        }
    expect_true(TRUE)
    })
})


