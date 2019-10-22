# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# zb17<-read.csv('../data/fieldData/treeRegeneration.csv')
zb17<-read.csv('../data/field/2017data/treeRegeneration.csv')
zb18<-read.csv('../data/field/2018data/treeRegeneration18.csv')

# zb18<-read.csv('./data/fieldData/treeRegeneration18.csv')
thresholds<-read.csv('../data/field/RF_thresholds.csv')

################
#Assign RF status
################
nrSites17<-c(4,7,11,12,20,23,37,43,44,53,56,58,64,66)
rSites17<-c(16,17,28,29,32,33,34,39,40,45,46,48,61,62)

zb17$RF<-2
for (sn in unique(zb17$site.number)){
  if (sn %in% nrSites17){zb17$RF[which(zb17$site.number == sn)] <- 1}
  if (sn %in% rSites17){zb17$RF[which(zb17$site.number == sn)] <- 0}
}

sDesc18<-read.csv('../data/field/2018data/siteDescriptions18.csv')
sDesc18$rcrtmnt[which(sDesc18$rcrtmnt == 'poor ')]<-'poor'
sDesc18$RF<-NA
sDesc18$RF[which(sDesc18$rcrtmnt == "abundant")]<-0
sDesc18$RF[which(sDesc18$rcrtmnt == "singular")]<-1
sDesc18$RF[which(sDesc18$rcrtmnt == "poor")]<-1
sDesc18$RF[which(sDesc18$rcrtmnt == "no regeneration")]<-1

zb18$RF<-NA
for (sn in unique(sDesc18$site.number)){
  zb18$RF[which(zb18$site.number == sn)]<-sDesc18$RF[which(sDesc18$site.number == sn)]
}

zb18$Vigor<-as.character(zb18$Vigor)
zb18$Vigor[which(zb18$Vigor == "heaithy")]<-"healthy"
zb18<-zb18[which(zb18$Vigor != "" & zb18$Vigor != "dead"),]

###########################
#consolidate plot categories
zb18$Plot.Format<-as.character(zb18$Plot.Format)
# returns string w/o leading or trailing whitespace
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

zb18$Plot.Format<-trim(zb18$Plot.Format)
zb18$Plot.Format[which(zb18$Plot.Format == 'seedlings per 2 m x 2 plot')]<-'seedlings per 2 m x 2 m plot'
zb18$Plot.Format[which(zb18$Plot.Format == 'seedlings per 2 x 2 m plot')]<-'seedlings per 2 m x 2 m plot'
zb18$Plot.Format[which(zb18$Plot.Format == 'seedlings per 2 m x 30 m')]<-'seedlings per 2 m x 30 m plot'
zb18$Plot.Format[which(zb18$Plot.Format == 'seedlings per 15-m radius plot')]<-'seedlings per 15 m radius plot'
zb18$Plot.Format[which(zb18$Plot.Format == 'seedlings per 15-m radius circle')]<-'seedlings per 15 m radius plot'

###########################
#Calculate stocking density
#Convert to tall seedlings
###########################
zb17$stocking.density<-NA
zb17$seedling.height.category..m.<-as.character(zb17$seedling.height.category..m.)
for (sn in unique(zb17$site.number)){
  RF2017<-unique(zb17$RF[which(zb17$site.number == sn)])
  sData<-zb17[which(zb17$site.number == sn),]
  #areaSubPlotsSqM<-length(unique(sData$transect.plot)) #1 sq meter quadrats
  areaSubPlotsSqMmodis<-10 #lack of obs == lack of regrowth (NOT 5!)
  areaSubPlotsSqMlandsat<-4
  
  shortSeedlings<-sData[which(zb17$seedling.height.category..m. == "1 yr sdlng"|sData$seedling.height.category..m. == "0.11-0.25" | sData$seedling.height.category..m. == "0.26-0.50"),]
  shortCount<-nrow(shortSeedlings[which(as.character(shortSeedlings$vigor) == "healthy"),]) + (0.5*nrow(shortSeedlings[which(as.character(shortSeedlings$vigor) == "weakened"),]))
  #NOT RIGHT, MEDIUM SHOULD BE 1 TO 1.5 NOT 2
  mediumSeedlings<-sData[which(zb17$seedling.height.category..m. == "gte 0.1"|sData$seedling.height.category..m. == "0.51-1.00" | sData$seedling.height.category..m. == "1.01-2.00"),]
  mediumCount<-nrow(mediumSeedlings[which(as.character(mediumSeedlings$vigor) == "healthy"),]) + (0.5*nrow(mediumSeedlings[which(as.character(mediumSeedlings$vigor) == "weakened"),]))
  tallSeedlings<-sData[which(sData$seedling.height.category..m. == "gt 2.01"),]
  tallCount<-nrow(tallSeedlings[which(as.character(tallSeedlings$vigor) == "healthy"),]) + (0.5*nrow(tallSeedlings[which(as.character(tallSeedlings$vigor) == "weakened"),]))
  sCountMODIS<-(shortCount*0.5)+(mediumCount*0.8)+tallCount
  
  sDataLs<-sData[which(sData$transect.plot<=30),]
  shortSeedlingsLs<-sDataLs[which(sDataLs$seedling.height.category..m. == "0.11-0.25" | sDataLs$seedling.height.category..m. == "0.26-0.50"),]
  shortCountLs<-nrow(shortSeedlingsLs[which(as.character(shortSeedlingsLs$vigor) == "healthy"),]) + (0.5*nrow(shortSeedlingsLs[which(as.character(shortSeedlingsLs$vigor) == "weakened"),]))
  #NOT RIGHT, MEDIUM SHOULD BE 1 TO 1.5 NOT 2
  mediumSeedlingsLs<-sDataLs[which(sDataLs$seedling.height.category..m. == "0.51-1.00" | sDataLs$seedling.height.category..m. == "1.01-2.00"),]
  mediumCountLs<-nrow(mediumSeedlingsLs[which(as.character(mediumSeedlingsLs$vigor) == "healthy"),]) + (0.5*nrow(mediumSeedlingsLs[which(as.character(mediumSeedlingsLs$vigor) == "weakened"),]))
  tallSeedlingsLs<-sDataLs[which(sDataLs$seedling.height.category..m. == "gt 2.01"),]
  tallCountLs<-nrow(tallSeedlingsLs[which(as.character(tallSeedlingsLs$vigor) == "healthy"),]) + (0.5*nrow(tallSeedlingsLs[which(as.character(tallSeedlingsLs$vigor) == "weakened"),]))
  sCountLs<-(shortCountLs*0.5)+(mediumCountLs*0.8)+tallCountLs
  
  sDens2017modis<-sCountMODIS/areaSubPlotsSqMmodis*10
  sDens2017Ls<-sCountLs/areaSubPlotsSqMlandsat*10
  
  densInfoPostFire<-data.frame(sn,RF2017,sDens2017modis,sDens2017Ls)
  if (exists('densInfoPostFireDf') == F){
    densInfoPostFireDf<-densInfoPostFire
  } else {
    densInfoPostFireDf<-rbind(densInfoPostFireDf,densInfoPostFire)
  }
}
densInfoPostFireDf$RF2018<-NA
densInfoPostFireDf$sDens2018<-NA

zb18$stocking.density<-NA
zb18$Hm<-as.character(zb18$Hm)
zb18$Hm[which(zb18$Hm == "broken" | zb18$Hm == "?" | zb18$Hm == "")]<-0
zb18$Hm<-as.numeric(zb18$Hm)
for (sn in unique(sDesc18$site.number)){
  RF2018<-unique(zb18$RF[which(zb18$site.number == sn)])
  sData<-zb18[which(zb18$site.number == sn),]
  if (nrow(sData) == 0){sDens1kPerHa <-0}
  if (nrow(sData) == 0){RF2018 <- 1}
  else{
    if (unique(sData$Plot.Format) == "seedlings per 2 m x 2 m plot"){
      areaSubPlotsSqM<-40 # 10 2x2 m plots
    }
    if (unique(sData$Plot.Format) == "seedlings per 2 m x 30 m plot"){
      areaSubPlotsSqM<-180 # 1 3x2x30 m plots
    }
    if (unique(sData$Plot.Format) == "seedlings per 15 m radius plot"){
      areaSubPlotsSqM<-706.5 # 1 15 m radius plots
    }
    
    #discounting "weakend"
    shortSeedlings<-sData[which(sData$Hm<=0.5),]
    shortCount<-nrow(shortSeedlings[which(as.character(shortSeedlings$Vigor) == "healthy"),]) + (0.5*nrow(shortSeedlings[which(as.character(shortSeedlings$Vigor) == "weakened"),]))
    mediumSeedlings<-sData[which(sData$Hm>0.5 & sData$Hm<1.5),]
    mediumCount<-nrow(mediumSeedlings[which(as.character(mediumSeedlings$Vigor) == "healthy"),]) + (0.5*nrow(mediumSeedlings[which(as.character(mediumSeedlings$Vigor) == "weakened"),]))
    tallSeedlings<-sData[which(sData$Hm>1.5),]
    tallCount<-nrow(tallSeedlings[which(as.character(tallSeedlings$Vigor) == "healthy"),]) + (0.5*nrow(tallSeedlings[which(as.character(tallSeedlings$Vigor) == "weakened"),]))
    #converting to tall seedlings
    sCount<-(shortCount*0.5)+(mediumCount*0.8)+tallCount
    sDens2018<-sCount/areaSubPlotsSqM*10
  }
  if (sn %in% densInfoPostFireDf$sn == F){
    RF2017<-NA
    sDens2017modis<-NA
    sDens2017Ls<-NA
    densInfoPostFireDf<-rbind(densInfoPostFireDf,data.frame(sn,RF2017,sDens2017modis,sDens2017Ls,RF2018,sDens2018))
  } else{
    densInfoPostFireDf$sDens2018[which(densInfoPostFireDf$sn == sn)]<-sDens2018
    densInfoPostFireDf$RF2018[which(densInfoPostFireDf$sn == sn)]<-RF2018
  }
}

###########################
#Compare w threshold data
###########################
densInfoPostFireDf$threshold<-NA
for (sn in unique(thresholds$Site[which(!is.na(thresholds$Site))])){
  threshold<-thresholds$Sufficient.regeneration.threshold..1k.ha.[which(thresholds$Site == sn)]
  densInfoPostFireDf$threshold[which(densInfoPostFireDf$sn == sn)] <- threshold
}

densInfoPostFireDf$fracThresh2017modis<-densInfoPostFireDf$sDens2017modis/densInfoPostFireDf$threshold
densInfoPostFireDf$fracThresh2017Ls<-densInfoPostFireDf$sDens2017Ls/densInfoPostFireDf$threshold
densInfoPostFireDf$fracThresh2018<-densInfoPostFireDf$sDens2018/densInfoPostFireDf$threshold

densInfoPostFireDf$RF2017 <- factor(densInfoPostFireDf$RF2017, levels = c('0','2','1'),ordered = TRUE)
print(ggplot(densInfoPostFireDf, aes(x=RF2017, y = fracThresh2017Ls, fill=RF2017)) + geom_boxplot()+
        scale_x_discrete(limits=c('0','2','1')) +
        scale_fill_manual(values=c("#636363","#969696","#cccccc")))

print(ggplot(densInfoPostFireDf, aes(x=RF2017, y = fracThresh2017modis, fill=RF2017)) + geom_boxplot()+
        scale_x_discrete(limits=c('0','2','1')) +
        scale_fill_manual(values=c("#636363","#969696","#cccccc")))

densInfoPostFireDf$RF2018 <- factor(densInfoPostFireDf$RF2018, levels = c('0','1'),ordered = TRUE)
print(ggplot(densInfoPostFireDf, aes(x=RF2018, y = fracThresh2018, fill=RF2018)) + geom_boxplot()+
        scale_x_discrete(limits=c('0','1')) +
        scale_fill_manual(values=c("#636363","#cccccc")))

browser()
