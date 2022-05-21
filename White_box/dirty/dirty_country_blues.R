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


#data generation - Country & Blues

country_data<- read_csv("country.csv")
country_data$binary_genre<-0

blues_data<-read_csv("blues.csv")
blues_data$binary_genre<-1


#adding dirty data
classical <- read_csv("classical.csv")
classical <- classical[1:10,]
classical$binary_genre<-0
classical$binary_genre[6:10]<-1

metal <- read_csv("metal.csv")
metal <- metal[1:10,]
metal$binary_genre<-0
metal$binary_genre[6:10]<-1

disco <- read_csv("disco.csv")
disco <- disco[1:10,]
disco$binary_genre<-0
disco$binary_genre[6:10]<-1

jazz <- read_csv("jazz.csv")
jazz <- jazz[1:10,]
jazz$binary_genre<-0
jazz$binary_genre[6:10]<-1

hiphop <- read_csv("hiphop.csv")
hiphop <- hiphop[1:10,]
hiphop$binary_genre<-0
hiphop$binary_genre[6:10]<-1

pop <- read_csv("pop.csv")
pop <- pop[1:10,]
pop$binary_genre<-0
pop$binary_genre[6:10]<-1

reggae <- read_csv("reggae.csv")
reggae <- reggae[1:10,]
reggae$binary_genre<-0
reggae$binary_genre[6:10]<-1

rock <- read_csv("rock.csv")
rock <- rock[1:10,]
rock$binary_genre<-0
rock$binary_genre[6:10]<-1

train_data<-rbind(country_data[0:80,],blues_data[0:80,])
test_data<-rbind(country_data[81:100,],blues_data[81:100,])


train_data<-rbind(train_data,classical,metal,disco,jazz,hiphop,pop,reggae,rock)

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
# 0.7333333  0.5


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
# 0.6916667  0.4


#logistic regression + covariate selection

glm_model<-glm(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast ,family=binomial( link = logit ), data = train_data)
summary(glm_model)
pscl::pR2(glm_model)["McFadden"]
# R^2 = 0.1806154


residualPlots(glm_model)
#outlier detection
out = which(abs(glm_model$residuals)/sd(glm_model$residuals) > 2)
out
train_data_clean = train_data[-out,]

#we iterate 6 times
glm_model<-glm(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast ,family=binomial( link = logit ), data = train_data_clean)
summary(glm_model)
residualPlots(glm_model)
out = which(abs(glm_model$residuals)/sd(glm_model$residuals) > 2)
out
train_data_clean = train_data_clean[-out,]

summary(glm_model)
pscl::pR2(glm_model)["McFadden"]
# R^2 = 0.8580381

which(train_data$mean_chroma %in% train_data_clean$mean_chroma)
#togheter whit some true outliers, we also deleted damaging true observations, outlier removal could be done as standard practice

# we apply backward selection: in order we remove hf_contrast, spec_flat, zrc, mf_contrast
glm_model_red <-glm(binary_genre~ rms_energy+mean_chroma+lf_contrast ,family=binomial( link = logit ), data = train_data_clean)
summary(glm_model_red)
pscl::pR2(glm_model_red)["McFadden"]
# R^2 = 0.8060218

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
# 0.942963  0.375