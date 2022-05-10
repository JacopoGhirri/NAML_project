setwd("D:/Marta/Politecnico/Numerical analysis for machine learning/Project/NAML_project/White_box/Dati")
#####
library(MASS)
library(class)

library(mvtnorm)
library(mvnormtest)
library(car)
library(MASS)
library(class)
library(readr)


library(rms)
library(arm)
library(ResourceSelection)
library(pROC)

mcshapiro.test <- function(X, devstmax = 0.01, sim = ceiling(1/(4*devstmax^2)))
{
  library(mvnormtest)
  n   <- dim(X)[1]
  p   <- dim(X)[2]
  mu  <- rep(0,p)
  sig <- diag(p)
  W   <- NULL
  for(i in 1:sim)
  {
    Xsim <- rmvnorm(n, mu, sig)
    W   <- c(W, mshapiro.test(t(Xsim))$stat)
    # mshapiro.test(X): compute the statistics min(W) for the sample X
  }
  Wmin   <- mshapiro.test(t(X))$stat   # min(W) for the given sample
  pvalue <- sum(W < Wmin)/sim          # proportion of min(W) more extreme than the observed Wmin
  devst  <- sqrt(pvalue*(1-pvalue)/sim)
  list(Wmin = as.vector(Wmin), pvalue = pvalue, devst = devst, sim = sim)
}
#####

#data generation - REGGAE & JAZZ

reggae_data<- read_csv("reggae.csv")
reggae_data$binary_genre<-0

jazz_data<-read_csv("jazz.csv")
jazz_data$binary_genre<-1

data<-rbind(reggae_data,jazz_data)
  


genre<-factor(data$binary_genre)
levels(genre)

#priors 
p<-c(1/2,1/2)

#assumptions for qda, lda

#gauss
mcshapiro.test(data[which(data$binary_genre=="1"),1:7]) 
mcshapiro.test(data[which(data$binary_genre=="0"),1:7])


#covariance 
v1<-var(data[which(data$binary_genre==1),1:7])
v2<-var(data[which(data$binary_genre==0),1:7])
v1
v2



#QDA (dati gaussiani, no same covariance)
q<-qda(data$binary_genre~ data$zcr+data$rms_energy+data$mean_chroma+data$spec_flat+data$hf_contrast+data$mf_contrast+data$lf_contrast  ,prior=p)
q #means

#aper
Qda.m <- predict(q)
f= factor(genre)
table(true.lable=f, class.assigned=Qda.m$class)

l <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Qda.m$class )
APER_qda <- 0
for(i in 1:l){
  APER_qda <- APER_qda + sum(t[i,-i])*p[i]/sum(t[i,])
}
APER_qda
#0.145

#LDA (dati NON gaussiani, same covariance)
l<-lda(data$binary_genre~ data$zcr+data$rms_energy+data$mean_chroma+data$spec_flat+data$hf_contrast+data$mf_contrast+data$lf_contrast  ,prior=p)
l #means

#aper
Lda.m <- predict(l)
f= factor(genre)
table(true.lable=f, class.assigned=Lda.m$class)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Lda.m$class )
APER_lda <- 0
for(i in 1:len){
  APER_lda <- APER_lda + sum(t[i,-i])*p[i]/sum(t[i,])
}
APER_lda
#0.125

#logistic regression + covariate selection

glm_model<-glm(data$binary_genre~ data$zcr+data$rms_energy+data$mean_chroma+data$spec_flat+data$hf_contrast+data$mf_contrast+data$lf_contrast ,family=binomial( link = logit ))
summary(glm_model)

glm_model_1<-glm(data$binary_genre~ data$zcr+data$rms_energy+data$mean_chroma+data$spec_flat+data$hf_contrast+data$mf_contrast ,family=binomial( link = logit ))
summary(glm_model_1)

glm_model_2<-glm(data$binary_genre~ data$rms_energy+data$mean_chroma+data$spec_flat+data$hf_contrast+data$mf_contrast ,family=binomial( link = logit ))
summary(glm_model_2)

glm_model_3<-glm(data$binary_genre~ data$mean_chroma+data$spec_flat+data$hf_contrast+data$mf_contrast ,family=binomial( link = logit ))
summary(glm_model_3)

glm_model_4<-glm(data$binary_genre~ data$mean_chroma+data$hf_contrast+data$mf_contrast ,family=binomial( link = logit ))
summary(glm_model_4)


anova( glm_model_4, glm_model, test = "Chisq" )
#if low pvalue recuced model is less significant

#table
threshold = 0.5
real  = data$binary_genre
predicted = as.numeric( glm_model_4$fitted.values > threshold )
# 1 se > soglia, 0 se < = soglia

tab = table( real, predicted )
tab

#% casi classificati correttamente 
accuracy=round( sum( diag( tab ) ) / sum( tab ), 2 )
accuracy
#0.88
1-accuracy

#% casi 1 classificati come 1 (classical)
sensitivity=tab [ 2, 2 ] /( tab [ 2, 1 ] + tab [ 2, 2 ] ) 
sensitivity
#0.87

#% casi 0 classificati come 0 (jazz)
specificity= tab[ 1, 1 ] /( tab [ 1, 2 ] + tab [ 1, 1 ] )
specificity
#0.88

