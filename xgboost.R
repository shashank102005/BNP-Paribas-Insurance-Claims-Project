
############# user defined functions
pmiss2 = function(x){sum(is.na(x))}

replace_missing = function(x){ 
  if(is.na(x))
      x = -999
}

##################### process the training data ##################################################

################ read the training data 
train_data = read.csv(file="F:/Projects/Kaggle/BNP Paribas/train.csv",header=TRUE, sep=",", na.strings=c(""))
train_data2 = train_data

############## remove id and target variables
train_data$id = NULL
train_data$target = NULL


############## count number of nas in each row and add as a new feature
na_train = numeric(nrow(train_data))
na_train  = apply(train_data, 1, function(z) sum(is.na(z)))
train_data$count_na = na_train 


############ count the number of nas in categorical and continuous vatiables
na_train_cont = numeric(nrow(train_data))

for(i in 1: length(na_train_cont))  
{ 
  sum =0
  for( j in 1:ncol(train_data))
  {
    if(is.na(train_data[i,j]) && (!is.factor(train_data[i,j])))
     sum = sum+1
  } 
  na_train_cont[i] = sum
}
train_data$count_na_cont = na_train_cont


na_train_cat = numeric(nrow(train_data))

for(i in 1: length(na_train_cat))  
{ 
  sum =0
  for( j in 1:ncol(train_data))
  {
    if(is.na(train_data[i,j]) && is.factor(train_data[i,j]))
      sum = sum+1
  } 
  na_train_cat[i] = sum
}
train_data$count_na_cat = na_train_cat


############# compute missingness matrix
miss_train = matrix(data = 1,nrow = nrow(train_data),ncol = ncol(train_data))
for(i in 1:nrow(train_data))
{
  for(j in 1:ncol(train_data))
  {
    if(is.na(train_data[i,j]))
       miss_train[i,j] = 0
  }
}

miss_train_df = as.data.frame(miss_train)


############## impute the missing data with -9999
train_data[is.na(train_data)] = -999


for (col in colnames(train_data)){
  
  if(is.factor(train_data[[col]]))
    
  {
    train_data[[col]] = factor(train_data[[col]], levels=c(levels(train_data[[col]]), 'XXXX')) 
    
  }
  
}

############ impute missing categorical data with 'XXX'
train_data[is.na(train_data)] = 'XXXX'


############### convert v22 into numeric 
train_data$v22 = factor(train_data$v22, labels=(1:length(levels(factor(train_data$v22)))))
train_data$v22= as.numeric(train_data$v22)


############ add missingness features to the train data set
train_data = cbind(train_data,miss_train_df)

##################################################################################################





##################################### process the test data  #####################################

############## read the testing data 
test_data = read.csv(file="F:/Projects/Kaggle/BNP Paribas/test.csv",header=TRUE, sep=",",
                      na.strings=c(""))
test_data2 = test_data

########## remove the id variable
test_data$id = NULL

############## count number of nas in each row and add as a new feature
na_test= numeric(nrow(test_data))
na_test  = apply(test_data, 1, function(z) sum(is.na(z)))
test_data$count_na = na_test 


############ count the number of nas in categorical and continuous vatiables

na_test_cont = numeric(nrow(test_data))

for(i in 1: length(na_test_cont))  
{ 
  sum =0
  for( j in 1:ncol(test_data))
  {
    if(is.na(test_data[i,j]) && (!is.factor(test_data[i,j])))
      sum = sum+1
  } 
  na_test_cont[i] = sum
}
test_data$count_na_cont = na_test_cont


na_test_cat = numeric(nrow(test_data))

for(i in 1: length(na_test_cat))  
{ 
  sum =0
  for( j in 1:ncol(test_data))
  {
    if(is.na(test_data[i,j]) && is.factor(test_data[i,j]))
      sum = sum+1
  } 
  na_test_cat[i] = sum
}
test_data$count_na_cat = na_test_cat


############# compute missingness matrix
miss_test = matrix(data = 1,nrow = nrow(test_data),ncol = ncol(test_data))
for(i in 1:nrow(test_data))
{
  for(j in 1:ncol(test_data))
  {
    if(is.na(test_data[i,j]))
      miss_test[i,j] = 0
  }
}

miss_test_df = as.data.frame(miss_test)


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



########### convert v22 into numeric 
test_data$v22 = factor(test_data$v22, labels=(1:length(levels(factor(test_data$v22)))))
test_data$v22= as.numeric(test_data$v22)


############ add missingness features to the test data set
test_data = cbind(test_data,miss_test_df)

################################################################################################



############################## one hot encoding ################################################
temp_df = rbind(train_data,test_data)
temp_df_ohe = dummy.data.frame(temp_df,dummy.classes = "factor")


train_ohe = temp_df_ohe[1:nrow(train_data),]
test_ohe = temp_df_ohe[(nrow(train_data)+1):nrow(temp_df_ohe),]



######################################### Run Cross Validation- XGBoost ####################################
#### cv train log loss: 0.384  test log loss:0.459
X_df= train_ohe
X_df = as.matrix(X_df)


Y_df = train_data2[,'target']
Y_df = as.numeric(Y_df)


bst.cv = xgb.cv (data = X_df, 
                 label = Y_df, 
                 eta = 0.01,
                 max_depth = 10, 
                 min_child_weight = 1,
                 nround=1288, 
                 subsample = 0.7,
                 colsample_bytree = 0.7,
                 eval_metric = "logloss",
                 objective = "binary:logistic",
                 verbose =1,
                 nfold = 5,
                 prediction = TRUE)


bst.cv$dt


################################### Final xgboost model for the test data #########################
## score on actual test data: 0.457 
final.xgb = xgboost(data = X_df, 
                    label = Y_df, 
                    eta = 0.01,
                    max_depth = 10, 
                    min_child_weight = 1,
                    nround=1288, 
                    subsample = 0.7,
                    colsample_bytree = 0.7,
                    eval_metric = "logloss",
                    objective = "binary:logistic",
                    verbose =1
                   )
############# performance on test data
X_test = test_ohe
X_test = as.matrix(X_test)

prob_test = predict(final.xgb, X_test)


########### feature importance
feat_imp = xgb.importance(colnames(X_df), model = final.xgb,data = )
xgb.plot.importance(importance_matrix = feat_imp)

################################################################################################



########## create a new data frame with only significant features
new_df = subset(train_ohe,select=feat_imp$Feature)
new_df_test = subset(test_ohe,select=feat_imp$Feature)




#################### Run Cross Validation- XGBoost on new filtered data frame  #################
#### cv train log loss: 0.312  test log loss:0.458
X_df= new_df
X_df = as.matrix(X_df)


Y_df = train_data2[,'target']
Y_df = as.numeric(Y_df)


bst.cv = xgb.cv (data = X_df, 
                 label = Y_df, 
                 eta = 0.009,
                 max_depth = 9, 
                 min_child_weight = 1,
                 nround=1600, 
                 subsample = 0.6,
                 colsample_bytree = 0.6,
                 eval_metric = "logloss",
                 objective = "binary:logistic",
                 verbose =1,
                 nfold = 5,
                 prediction = TRUE)



### actual score on test data : 0.45709 
final.xgb = xgboost(data = X_df, 
                    label = Y_df, 
                    eta = 0.01,
                    max_depth = 10, 
                    min_child_weight = 1,
                    nround=1288, 
                    subsample = 0.7,
                    colsample_bytree = 0.7,
                    eval_metric = "logloss",
                    objective = "binary:logistic",
                    verbose =1
                     )


########### feature importance
feat_imp2 = xgb.importance(colnames(X_df), model = final.xgb )
xgb.plot.importance(importance_matrix = feat_imp2)

X_test = new_df_test
X_test = as.matrix(X_test)

prob_test = predict(final.xgb, X_test)





########## create a new data frame with only significant features - further filter
new_df2 = subset(train_ohe,select=feat_imp2$Feature[1:325])
new_df_test2 = subset(test_ohe,select=feat_imp2$Feature[1:325])




#################### Run Cross Validation- XGBoost on new filtered data frame  #################
#### cv train log loss: 0.3094  test log loss:0.4587
X_df= new_df2
X_df = as.matrix(X_df)


Y_df = train_data2[,'target']
Y_df = as.numeric(Y_df)


bst.cv = xgb.cv (data = X_df, 
                 label = Y_df, 
                 eta = 0.01,
                 max_depth = 10, 
                 min_child_weight = 1,
                 nround=1288, 
                 subsample = 0.7,
                 colsample_bytree = 0.7,
                 eval_metric = "logloss",
                 objective = "binary:logistic",
                 verbose =1,
                 nfold = 5,
                 prediction = TRUE)




final.xgb = xgboost(data = X_df, 
                    label = Y_df, 
                    eta = 0.01,
                    max_depth = 10, 
                    min_child_weight = 1,
                    nround=1288, 
                    subsample = 0.7,
                    colsample_bytree = 0.7,
                    eval_metric = "logloss",
                    objective = "binary:logistic",
                    verbose =1
                    )


########### feature importance
feat_imp2 = xgb.importance(colnames(X_df), model = final.xgb )
xgb.plot.importance(importance_matrix = feat_imp2)

X_test = new_df_test2
X_test = as.matrix(X_test)

prob_test = predict(final.xgb, X_test)

################################################################################################





############################ Extracting new features from the missingness matrix ###############

########tetrachoric correlation matrix
poly_mat = tetrachoric(x = miss_train_df,y=NULL,correct=TRUE,smooth=TRUE,global=TRUE,weight=NULL,
                       na.rm=TRUE, delete=TRUE)

poly_mat_test = tetrachoric(x = miss_test_df,y=NULL,correct=TRUE,smooth=TRUE,global=TRUE,weight=NULL,
                       na.rm=TRUE, delete=TRUE)

########PCA with tetrachoric correlation matrix
princ_comp_tetra = princomp(poly_mat$rho,cor=TRUE) 
summary(princ_comp_tetra, loading=FALSE)
plot(princ_comp_tetra,type="lines") ### scree plot

princ_comp_tetra_test = princomp(poly_mat_test$rho,cor=TRUE)
prin_score_test = princ_comp_tetra_test$scores

####### extract first two principal components (99 % of variance explained)
comp_mat = princ_comp_tetra$scores[,1:2]
miss_train = subset(miss_train_df,select = rownames(prin_score))
miss_comp = as.matrix(miss_train) %*% comp_mat

comp_mat_test = princ_comp_tetra_test$scores[,1:2]
miss_test = subset(miss_test_df,select = rownames(prin_score_test))
miss_comp_test = as.matrix(miss_test) %*% comp_mat_test


###### create new data frame with these features
new_df3 = cbind(new_df2,miss_comp)



X_df= new_df3
X_df = as.matrix(X_df)


Y_df = train_data2[,'target']
Y_df = as.numeric(Y_df)

## cv train: 0.3066 , test: 0.4854
bst.cv = xgb.cv (data = X_df, 
                 label = Y_df, 
                 eta = 0.01,
                 max_depth = 10, 
                 min_child_weight = 1,
                 nround=1329, 
                 subsample = 0.7,
                 colsample_bytree = 0.7,
                 eval_metric = "logloss",
                 objective = "binary:logistic",
                 verbose =1,
                 nfold = 5,
                 prediction = TRUE)


final.xgb = xgboost(data = X_df, 
                    label = Y_df, 
                    eta = 0.01,
                    max_depth = 10, 
                    min_child_weight = 1,
                    nround=1329, 
                    subsample = 0.7,
                    colsample_bytree = 0.7,
                    eval_metric = "logloss",
                    objective = "binary:logistic",
                    verbose =1
                    )

########### feature importance
feat_imp3 = xgb.importance(colnames(X_df), model = final.xgb )
xgb.plot.importance(importance_matrix = feat_imp2)

X_test = cbind(new_df_test2,miss_comp_test)
X_test = as.matrix(X_test)

prob_test = predict(final.xgb, X_test)




########################## submit results
submit_bnp = data.frame(test_data2$id,prob_test)
names(submit_bnp)[1] = paste("ID")
names(submit_bnp)[2] = paste("PredictedProb")

write.csv(submit_bnp,"submit_bnp.csv",row.names = FALSE)


write.csv(train_ohe,"train_ohe.csv",row.names = FALSE)
write.csv(test_ohe,"test_ohe.csv",row.names = FALSE)





