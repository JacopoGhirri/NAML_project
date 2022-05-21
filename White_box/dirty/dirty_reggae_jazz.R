setwd("C:/Users/user/Desktop/università/dare/Numerical Analysis for Machine Learning/NAML proj/NAML_repo/NAML_project/White_box/Dati")
#####
library(MASS)
library(class)

library(mvtnorm)
library(mvnormtest)
library(car)
library(MASS)
library(class)
library(readr)
library(pscl)


library(rms)
library(arm)
library(ResourceSelection)
library(pROC)

#####

#data generation - REGGAE & JAZZ

reggae_data<- read_csv("reggae.csv")
reggae_data$binary_genre<-0

jazz_data<-read_csv("jazz.csv")
jazz_data$binary_genre<-1


#adding dirty data
blues <- read_csv("blues.csv")
blues <- blues[1:10,]
blues$binary_genre<-0
blues$binary_genre[6:10]<-1

country <- read_csv("country.csv")
country <- country[1:10,]
country$binary_genre<-0
country$binary_genre[6:10]<-1

disco <- read_csv("disco.csv")
disco <- disco[1:10,]
disco$binary_genre<-0
disco$binary_genre[6:10]<-1

metal <- read_csv("metal.csv")
metal <- metal[1:10,]
metal$binary_genre<-0
metal$binary_genre[6:10]<-1

hiphop <- read_csv("hiphop.csv")
hiphop <- hiphop[1:10,]
hiphop$binary_genre<-0
hiphop$binary_genre[6:10]<-1

pop <- read_csv("pop.csv")
pop <- pop[1:10,]
pop$binary_genre<-0
pop$binary_genre[6:10]<-1

classical <- read_csv("classical.csv")
classical <- classical[1:10,]
classical$binary_genre<-0
classical$binary_genre[6:10]<-1

rock <- read_csv("rock.csv")
rock <- rock[1:10,]
rock$binary_genre<-0
rock$binary_genre[6:10]<-1

train_data<-rbind(reggae_data[0:80,],jazz_data[0:80,])
test_data<-rbind(reggae_data[81:100,],jazz_data[81:100,])

train_data<-rbind(train_data,blues,country,disco,classical,hiphop,pop,metal,rock)

#priors 
p<-c(1/2,1/2)

#assumptions for qda, lda are pointless since one third of the dataset is composed of outliers


#QDA (dati gaussiani, no same covariance)
q<-qda(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, prior=p, data = train_data)
q #means

#aper_train data
Qda.m <- predict(object=q, method = "plug-in")
f= factor(train_data$binary_genre)
table(true.lable=f, class.assigned=Qda.m$class)

l <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Qda.m$class )
train_APER_qda <- 0
for(i in 1:l){
  train_APER_qda <- train_APER_qda + sum(t[i,-i])*p[i]/sum(t[i,])
}
train_APER_qda

#aper_test data
Qda.m <- predict(object = q, newdata = data.frame(test_data[,1:7]), method = "plug-in")
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=Qda.m$class)

l <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Qda.m$class )
test_APER_qda <- 0
for(i in 1:l){
  test_APER_qda <- test_APER_qda + sum(t[i,-i])*p[i]/sum(t[i,])
}
test_APER_qda

qda_accuracies = cbind(training = 1-train_APER_qda,test = 1-test_APER_qda)
qda_accuracies
# training  test
# 0.7708333 0.65


#LDA (dati NON gaussiani, same covariance)
l<-lda(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast,prior=p, data = train_data)
l #means

#aper_train data
Lda.m <- predict(l, method = "plug-in")
f= factor(train_data$binary_genre)
table(true.lable=f, class.assigned=Lda.m$class)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Lda.m$class )
train_APER_lda <- 0
for(i in 1:len){
  train_APER_lda <- train_APER_lda + sum(t[i,-i])*p[i]/sum(t[i,])
}
train_APER_lda

#aper_test data
Lda.m <- predict(object = l,newdata = test_data, method = "plug-in")
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=Lda.m$class)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Lda.m$class )
test_APER_lda <- 0
for(i in 1:len){
  test_APER_lda <- test_APER_lda + sum(t[i,-i])*p[i]/sum(t[i,])
}
test_APER_lda

lda_accuracies = cbind(training = 1-train_APER_lda,test = 1-test_APER_lda)
lda_accuracies
# training  test
# 0.7541667 0.725

#relatively good performances

#logistic regression + covariate selection

glm_model<-glm(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast ,family=binomial( link = logit ), data = train_data)
summary(glm_model)
pscl::pR2(glm_model)["McFadden"]
# R^2 = 0.223769 

residualPlots(glm_model)
#outlier detection
out = which(abs(glm_model$residuals)/sd(glm_model$residuals) > 2)
out
train_data_clean = train_data[-out,]

#we iterate 2 times
glm_model<-glm(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast ,family=binomial( link = logit ), data = train_data_clean)
summary(glm_model)
residualPlots(glm_model)
out = which(abs(glm_model$residuals)/sd(glm_model$residuals) > 2)
out
train_data_clean = train_data_clean[-out,]


summary(glm_model)
pscl::pR2(glm_model)["McFadden"]
# R^2 = 0.4977484 

which(train_data$mean_chroma %in% train_data_clean$mean_chroma)
#indeed we eliminated the most damaging outliers

# we apply backward selection: in order we remove lf_contrast, mf_contrast

glm_model_red <-glm(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast,family=binomial( link = logit ), data = train_data_clean)
summary(glm_model_red)
pscl::pR2(glm_model_red)["McFadden"]
# R^2 = 0.4918645 

#accuracy on training data:
pred_train <- as.numeric(fitted(object = glm_model_red)>0.5)
f= factor(train_data_clean$binary_genre)
table(true.lable=f, class.assigned=pred_train)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =pred_train )
train_acc_logit <- 0
for(i in 1:len){
  train_acc_logit <- train_acc_logit + sum(t[i,i]*p[i])/sum(t[i,])
}
train_acc_logit
#accuracy on test data:
pred_test <- as.numeric(predict(object = glm_model_red, newdata= test_data, type="response")>0.5)
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=pred_test)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =pred_test )
test_acc_logit <- 0
for(i in 1:len){
  test_acc_logit <- test_acc_logit + sum(t[i,i]*p[i])/sum(t[i,])
}
test_acc_logit

logistic_regression_accuracies = cbind(training = train_acc_logit, test = test_acc_logit)
logistic_regression_accuracies
# training  test
# 0.8277769 0.675