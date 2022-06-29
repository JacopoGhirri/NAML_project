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

source("C:/Users/user/Desktop/università/dare/Numerical Analysis for Machine Learning/NAML proj/NAML_repo/NAML_project/White_box/metric_extractor.R")


#####

#data generation - CLASSICAL & METAL

classical_data<- read_csv("classical.csv")
classical_data$binary_genre<-0

metal_data<-read_csv("metal.csv")
metal_data$binary_genre<-1


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

train_data<-rbind(classical_data[0:80,],metal_data[0:80,])
test_data<-rbind(classical_data[81:100,],metal_data[81:100,])

train_data<-rbind(train_data,blues,country,disco,jazz,hiphop,pop,reggae,rock)

#priors 
p<-c(1/2,1/2)

#assumptions for qda, lda are pointless since one third of the dataset is composed of outliers


#QDA (dati gaussiani, no same covariance)
q<-qda(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, prior=p, data = train_data)
q #means

#metrics
Qda.m <- predict(object=q, method = "plug-in")
f= factor(train_data$binary_genre)
table(true.lable=f, class.assigned=Qda.m$class)

t_train <- table(true.label = f , assigned.label =Qda.m$class )

Qda.m <- predict(object = q, newdata = data.frame(test_data[,1:7]), method = "plug-in")
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=Qda.m$class)

t_test <- table(true.label = f , assigned.label =Qda.m$class )

qda_metrics <- get_metrics_train_test(t_train, t_test, 2)
qda_metrics

#           training test
# accuracy  0.8250000    1
# precision 0.8264509    1
# recall    0.8250000    1
# F1_score  0.8248053    1


#LDA (dati NON gaussiani, same covariance)
l<-lda(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast,prior=p, data = train_data)
l #means

#metrics
Lda.m <- predict(l, method = "plug-in")
f= factor(train_data$binary_genre)
table(true.lable=f, class.assigned=Lda.m$class)

t_train <- table(true.label = f , assigned.label =Lda.m$class )

Lda.m <- predict(object = l,newdata = test_data, method = "plug-in")
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=Lda.m$class)

t_test <- table(true.label = f , assigned.label =Lda.m$class )

lda_metrics <- get_metrics_train_test(t_train, t_test, 2)
lda_metrics

#            training test
# accuracy  0.8625000    1
# precision 0.8625252    1
# recall    0.8625000    1
# F1_score  0.8624976    1


#logistic regression + covariate selection

glm_model<-glm(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast ,family=binomial( link = logit ), data = train_data)
summary(glm_model)
pscl::pR2(glm_model)["McFadden"]
# R^2 = 0.5349397



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
# R^2 = 0.7843614 

which(train_data$mean_chroma %in% train_data_clean$mean_chroma)
#indeed we eliminated the most damaging outliers

# we apply backward selection: in order we remove spec_flat, rms_energy

glm_model_red <-glm(binary_genre~ zcr+mean_chroma+hf_contrast+mf_contrast+lf_contrast ,family=binomial( link = logit ), data = train_data_clean)
summary(glm_model_red)
pscl::pR2(glm_model_red)["McFadden"]
# R^2 = 0.7772986


#metrics
pred_train <- as.numeric(fitted(object = glm_model_red)>0.5)
f= factor(train_data_clean$binary_genre)
table(true.lable=f, class.assigned=pred_train)

t_train <- table(true.label = f , assigned.label =pred_train )

pred_test <- as.numeric(predict(object = glm_model_red, newdata= test_data, type="response")>0.5)
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=pred_test)

t_test <- table(true.label = f , assigned.label =pred_test )

LR_metrics <- get_metrics_train_test(t_train, t_test, 2)
LR_metrics

#            training      test
# accuracy  0.9244444 0.9750000
# precision 0.9244903 0.9761905
# recall    0.9244232 0.9750000
# F1_score  0.9244385 0.9749844