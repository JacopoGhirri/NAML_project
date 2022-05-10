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

#data generation

jazz_data<- read_csv("jazz.csv")
jazz_data$genre<-"jazz"

blues_data<-read_csv("blues.csv")
blues_data$genre<-"blues"

reggae_data<-read_csv("reggae.csv")
reggae_data$genre<-"reggae"

data<-rbind(jazz_data,blues_data,reggae_data)
  
genre<-factor(data$genre)
levels(genre)


#priors 
p<-rep(1/3,3)

#assumptions

#gauss
mcshapiro.test(data[which(genre=="blues"),1:7])
mcshapiro.test(data[which(genre=="jazz"),1:7])
mcshapiro.test(data[which(genre=="reggae"),1:7])

#covariance 
v1<-var(data[which(genre=="blues"),1:7])
v2<-var(data[which(genre=="jazz"),1:7])
v3<-var(data[which(genre=="reggae"),1:7])
v1
v2
v3


#QDA (dati gaussiani, no same covariance)
q<-qda(data$genre~ data$zcr+data$rms_energy+data$mean_chroma+data$spec_flat+data$hf_contrast+data$mf_contrast+data$lf_contrast ,prior=p)
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
#0.28333


#LDA (dati NON gaussiani, same covariance)
l<-lda(data$genre~ data$zcr+data$rms_energy+data$mean_chroma+data$spec_flat+data$hf_contrast+data$mf_contrast+data$lf_contrast  ,prior=p)
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
#0.27333

