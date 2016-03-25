
############# user defined functions
pmiss2 = function(x){sum(is.na(x))}

replace_missing = function(x){ 
  if(is.na(x))
      x = -999
}

##################### process the training data ################################################
################ read the training data 
train_data = read.csv(file="/media/shashank/Data/Projects/Kaggle/BNP Paribas/train.csv",
                      header=TRUE, sep=",", na.strings=c(""))

############## count number of nas in each row and add as a new feature
na_train = numeric(nrow(train_data))
na_train  = apply(train_data, 1, function(z) sum(is.na(z)))
train_data$count_na = na_train 


############## impute the missing data with -9999
train_data[is.na(train_data)] = -999


for (col in colnames(train_data)){
  
  if(is.factor(train_data[[col]]))
    
  {
    train_data[[col]] = factor(train_data[[col]], levels=c(levels(train_data[[col]]), 'XXXX')) 
    
  }
  
}

train_data[is.na(train_data)] = 'XXXX'

apply(train_data,2,pmiss2)

############### convert v22 into numeric 
train_data$v22 = factor(train_data$v22, labels=(1:length(levels(factor(train_data$v22)))))
train_data$v22= as.numeric(train_data$v22)

write.csv(train_data,"df_fixed_value_imputed.csv")

####################################################################################################



##################################### process the test data  #######################################
############## read the testing data 
test_data = read.csv(file="/media/shashank/Data/Projects/Kaggle/BNP Paribas/test.csv",header=TRUE, sep=",",
                      na.strings=c(""))


############## count number of nas in each row and add as a new feature
na_test= numeric(nrow(test_data))
na_test  = apply(test_data, 1, function(z) sum(is.na(z)))
test_data$count_na = na_test 


############## impute the missing continuous data with -9999
test_data[is.na(test_data)] = -999


for (col in colnames(test_data)){
  
  if(is.factor(test_data[[col]]))
    
  {
    test_data[[col]] = factor(test_data[[col]], levels=c(levels(test_data[[col]]), 'XXXX')) 
    
  }
  
}
############ impute missing categorical data with 'XXX'
test_data[is.na(test_data)] = 'XXXX'

apply(test_data,2,pmiss2)

########### convert v22 into numeric 
test_data$v22 = factor(test_data$v22, labels=(1:length(levels(factor(test_data$v22)))))
test_data$v22= as.numeric(test_data$v22)

write.csv(test_data,"test_fixed_value_imputed.csv")

################################################################################################





############ read the one hot encoded  train data
df_data = read.csv(file="/media/shashank/Data/Projects/Kaggle/BNP Paribas/fixed value imputed data/
                   train_onehot.csv",header=TRUE, sep=",",na.strings=c(""))

df_data$X = NULL
df_data$target = as.factor(as.numeric(train_data$target)) 

########### read the one hot encoded  test data
df_test= read.csv(file="/media/shashank/Data/Projects/Kaggle/BNP Paribas/fixed value imputed data/
                  test_onehot.csv",header=TRUE, sep=",",na.strings=c(""))
str(df_data)
df_test$X = NULL



##### divide into training and validation sets
samp = sample(nrow(df_data),nrow(df_data)*0.30)
df_train = df_data[-samp,]
df_val = df_data[samp,]


################################################ prepare data sets for xgboost
X_train = df_train
X_train$id = NULL
X_train$target = NULL
X_train = as.matrix(X_train)
Y_train = df_train[,'target']
Y_train = as.numeric(Y_train)

for(i in 1:length(Y_train))
{
  if(Y_train[i] == 1)
    Y_train[i]=0
  else
    Y_train[i]=1
}

X_val = df_val
X_val$id =  NULL
X_val$target =  NULL
X_val = as.matrix(X_val)
Y_val = df_val[,'target']
Y_val = as.numeric(Y_val)

for(i in 1:length(Y_val))
{
  if(Y_val[i] == 1)
    Y_val[i]=0
  else
    Y_val[i]=1
}
###########################################################################################


################### XG Boost Algorithm ####################################################
xgb  =      xgboost(data = X_train, 
                    label = Y_train, 
                    eta = 0.007,
                    max_depth = 6, 
                    min_child_weight = 1,
                    nround=3700, 
                    subsample = 0.6,
                    colsample_bytree = 0.5,
                    eval_metric = "logloss",
                    objective = "binary:logistic",
                    verbose =1)

############## performance on the validation set
prob = predict(xgb, X_val)
str(prob)


roc.plot(Y_val == '1',prob)$roc.vol

logloss = logLoss(Y_val, prob)
##########################################################################################




######################################### Run Cross Validation ####################################

X_df = as.matrix(df_data)

Y_df = train_data[,'target']
Y_df = as.numeric(Y_df)


bst.cv = xgb.cv (data = X_df, 
                 label = Y_df, 
                 eta = 0.007,
                 max_depth = 6, 
                 min_child_weight = 1,
                 nround=3700, 
                 subsample = 0.6,
                 colsample_bytree = 0.5,
                 eval_metric = "logloss",
                 objective = "binary:logistic",
                 verbose =1,
                 nfold = 5,
                 prediction = TRUE)


bst.cv$dt
