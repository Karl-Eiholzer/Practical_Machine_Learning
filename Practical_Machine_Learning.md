# Coursera Peer Reviewed Project - Practical Machine Learning Week 4 - Using accelerometers to predict if exercise is being performed correctly
Karl Eiholzer  
31 December 2016  
<h5 id="TOC">Table of Contents</h5>
<ul>
<li><a href="#Overview">Executive Summary</a></li>
<li><a href="#Prelims">Preliminary Steps</a>
<li><a href="#Step1">Exploratory Data Analysis</a>
<li><a href="#Step2">Data Cleaning</a>
<li><a href="#Step3">Fitting Multiple Prediction Models</a>
<li><a href="#Step4">Simplifying the Random Forest Model</a>
<li><a href="#Step5">Conclusions</a>
<li><a href="#App1">Appendix 1: Timestamps and Data</a>
<li><a href="#App2">Appendix 2: Random Forest Data</a>
<li><a href="#App2">Appendix 3: Simplified Random Forest Data</a>
<li><a href="#SysVers">System and Version Information</a>
</ul></li>
<h3 id="Overview">Executive Summary</h3>
<p align="right">Skip to <a href="#top">Top</a> or <a href="#Prelims">next section</a></p>

The goal of the project is to predict if a person is exercising correctly based on accelerometer measurements:<br>
<li>Lorem ipsum dolor sit amet, consectetur adipiscing elit. </li>
<li>Lorem ipsum dolor sit amet, consectetur adipiscing elit. </li>
<br>
<h5 id="Prelims">Preliminaries: load libraries we will be using</h5>
<p align="right">Skip to <a href="#Overview">prior section</a> or <a href="#Step1">next section</a></p></p>
Loading ggplot2 and caret packages, setting seed and loading in the data files.



<h3 id="Step1">Exploratory Data Analysis</h3>
<p align="right">Skip to <a href="#Prelims">prior section</a> or <a href="#Step2">next section</a></p>

The data comes from six test subjects (adelmo, carlitos, charles, eurico, jeremy, and pedro) engaged in five activities (labelled A thru E). As described by the original authors:
"Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg)." Accerlometer measurements were captured as the participants exercised.
[Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4TZMpuEj6]
<br>

We see a well distributed number of observations for each person for each activity, from 469 to 1,177: 

```r
with(raw.training, table(classe, user_name))
```

```
##       user_name
## classe adelmo carlitos charles eurico jeremy pedro
##      A   1165      834     899    865   1177   640
##      B    776      690     745    592    489   505
##      C    750      493     539    489    652   499
##      D    515      486     642    582    522   469
##      E    686      609     711    542    562   497
```

Many of the columns appear to be mostly comprised of NA values. The distribution of columns with high rates of NA values suggests on 59 columns are useful as predictors: 


```r
NA.rates <- cbind(Column = names(raw.training), Percent.NA = 0, High.NAs = FALSE)
y <- nrow(raw.training)
for(i in 1:length(raw.training) ) 
  { x <-  round( sum( is.na( raw.training[, i] ) ) / y * 100 , digits = 0 )
    NA.rates[i , 2 ] <- x
    if (x > 95) 
      { NA.rates[i , 3 ] <- TRUE
      }
}
table(NA.rates[,2])
```

```
## 
##   0 100  98 
##  60   6  94
```
Of the 60 columns contain zero NA values, while 100 columns contain 98 or 100 percent NA values. 

Also we see that the training data is organized by timestamp. The individuals worked through the various activities from A to E in order each time. Given that we would not normally have access to that information when predicting the activity, we want to build a model that is not dependent on timestamps. Finally we also see that the data is ordered by row number ("X"), which we would also not expect to occur in a normal environment:<br>

![](Practical_Machine_Learning_files/figure-html/Print Graphs Showing Timestampe Relationships-1.png)<!-- -->

See <a href="#App1">Appendix 1</a> for more data on this topic.


```
## [1] FALSE
```

<h3 id="Step2">Data Cleaning</h3>
<p align="right">Skip to <a href="#Step1">prior section</a> or <a href="#Step3">next section</a></p>

I'll remove the columns that contain mostly NA values and also any columns with near zero variance:

```r
sig.training <- raw.training
for(i in 1:length(raw.training)) 
    { if( sum( is.na( raw.training[, i] ) ) /nrow(raw.training) >= .95) 
      { for(j in 1:length(sig.training)) 
        { if( length( grep(names(raw.training[i]), names(sig.training)[j]) ) == 1)  
          { sig.training <- sig.training[ , -j]
          }   
        } 
      }
    }

col.no <- which(names(sig.training)=="classe")
nzv <- nearZeroVar(sig.training[, -col.no], 
                  freqCut = 99/1, 
                  uniqueCut = 5,
                  saveMetrics=TRUE)
nzv.training <- sig.training[,nzv$nzv==FALSE]

dim(nzv.training)
```

```
## [1] 19622    60
```

The near zero variance test does not remove any columns, indicating that any of the remaining columns may be useful for prediction.<br>

Lastly we will remove the user names and timestamp columns, as I believe they will not be useful for prediction. I also shuffle the rows randomly to assist with sampling techniques when fitting and cross validating. 


```r
col.no <- which(names(nzv.training)%in%c("user_name","new_window","X","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp"))
clean.training <- nzv.training[, -col.no]
clean.training <- clean.training[sample(nrow(clean.training)),]
# Clean up the objects
rm(nzv.training)
rm(sig.training)
```

<br>
<h3 id="Step3">Fitting a Random Forest Prediction Models</h3>
<p align="right">Skip to <a href="#Step2">prior section</a> or <a href="#Step4">next section</a></p>

After testing kNN, gbm and random forest models, I found the highest accuracy useing random forest. First I train a model using all available predictors.
<br>

I start by creating separate training and validation stes:

```r
set.seed(102767)
TrainIndex <- createDataPartition(clean.training$classe, 
                                  p = .7,        
                                  list = FALSE,  
                                  times = 2)     
Training <- clean.training[TrainIndex,]
Validation <- clean.training[-TrainIndex,]
```

Due to mimitation of the machine I am using, I will run 100 trees, instead of the recomended 400 or more.



Full informatiomn about the random forest fit may be found in  <a href="#App2">Appendix 2</a> :

```
## [1] FALSE
```

![](Practical_Machine_Learning_files/figure-html/Write log describing random forest prediction model-1.png)<!-- -->

This analysis identifies the most important variables for predicting which class of acticiy the exerciser was engaged in:

```r
rownames(importance(rf.fit)[1:25,0])
```

```
##  [1] "num_window"       "roll_belt"        "pitch_belt"      
##  [4] "yaw_belt"         "total_accel_belt" "gyros_belt_x"    
##  [7] "gyros_belt_y"     "gyros_belt_z"     "accel_belt_x"    
## [10] "accel_belt_y"     "accel_belt_z"     "magnet_belt_x"   
## [13] "magnet_belt_y"    "magnet_belt_z"    "roll_arm"        
## [16] "pitch_arm"        "yaw_arm"          "total_accel_arm" 
## [19] "gyros_arm_x"      "gyros_arm_y"      "gyros_arm_z"     
## [22] "accel_arm_x"      "accel_arm_y"      "accel_arm_z"     
## [25] "magnet_arm_x"
```
My next goal will be to create a simpler model with just those predictors.


<br>
<h3 id="Step4">Simplifying the Random Forest Model</h3>
<p align="right">Skip to <a href="#Step2">prior section</a> or <a href="#Step4">next section</a></p>

After testing a few smaller sets of predictors, I found that 15 variables would still achieve over 99% accuracy. I also found that accuracy improvements were not found after more than 60 trees. This allows me to run a more simplified version of the model:

```r
set.seed(102767)
key.var <- rownames(importance(rf.fit)[1:15,0])
rf.fit2 <- randomForest(classe ~ ., 
                       data = Training[,c("classe",key.var)],
                       ntree=60,
                       importance = TRUE,
                       proximity = TRUE  )
rf.result2 <- predict(rf.fit2, Validation[,c("classe",key.var)])
```

Full informatiomn about the simplified model may be found in  <a href="#App2">Appendix 3</a> :

```
## [1] FALSE
```

![](Practical_Machine_Learning_files/figure-html/Write log describing simplified random forest prediction model-1.png)<!-- -->


<h3 id="Step5">Conclusions</h3>
<p align="right">Skip to <a href="#Step3">prior section</a> or <a href="#SysVers">next section</a></p>

To confirm that the simplified model is as effective as the original full model, we can test both models against the testing data. :


```r
# make columns identical between training and testing sets
col.no <- which(names(testing)%in%as.vector(colnames(Training)))
final.testing1 <- testing[,col.no]
# make columns identical between training and testing sets
col.no <- which(names(testing)%in%as.vector(c("classe",key.var)))
final.testing2 <- testing[,col.no]
# predict results
final.result1 <- predict(rf.fit,  newdata = final.testing1)
final.result2 <- predict(rf.fit2, newdata = final.testing2)
```

A confusion matrix shows that both models return the same results:

```r
confusionMatrix(final.result1, final.result2)$table
```

```
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 0 8 0 0 0
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 0 0 3
```

Using the simplified model, the final predictions for the test set are as follows:

```r
final.result2
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

<br>
<h4 id="App1">Appendix 1: Timestamps and Data</h4>


```r
con <- file("Documentation/Exercise_Timestamp_data.txt", "r", blocking = FALSE)
readLines("Documentation/Exercise_Timestamp_data.txt")
```

```
##   [1] "##################################################### "                           
##   [2] "# This is a Log Decsribing Relationships between timestamps and users in the data"
##   [3] "# Created:  Mon Jan  2 08:02:03 2017 "                                            
##   [4] "##################################################### "                           
##   [5] " "                                                                                
##   [6] " "                                                                                
##   [7] ""                                                                                 
##   [8] ", , user_name = adelmo"                                                           
##   [9] ""                                                                                 
##  [10] "                  classe"                                                         
##  [11] "cvtd_timestamp       A   B   C   D   E"                                           
##  [12] "  02/12/2011 13:32 177   0   0   0   0"                                           
##  [13] "  02/12/2011 13:33 988 333   0   0   0"                                           
##  [14] "  02/12/2011 13:34   0 443 750 182   0"                                           
##  [15] "  02/12/2011 13:35   0   0   0 333 686"                                           
##  [16] "  02/12/2011 14:56   0   0   0   0   0"                                           
##  [17] "  02/12/2011 14:57   0   0   0   0   0"                                           
##  [18] "  02/12/2011 14:58   0   0   0   0   0"                                           
##  [19] "  02/12/2011 14:59   0   0   0   0   0"                                           
##  [20] "  05/12/2011 11:23   0   0   0   0   0"                                           
##  [21] "  05/12/2011 11:24   0   0   0   0   0"                                           
##  [22] "  05/12/2011 11:25   0   0   0   0   0"                                           
##  [23] "  05/12/2011 14:22   0   0   0   0   0"                                           
##  [24] "  05/12/2011 14:23   0   0   0   0   0"                                           
##  [25] "  05/12/2011 14:24   0   0   0   0   0"                                           
##  [26] "  28/11/2011 14:13   0   0   0   0   0"                                           
##  [27] "  28/11/2011 14:14   0   0   0   0   0"                                           
##  [28] "  28/11/2011 14:15   0   0   0   0   0"                                           
##  [29] "  30/11/2011 17:10   0   0   0   0   0"                                           
##  [30] "  30/11/2011 17:11   0   0   0   0   0"                                           
##  [31] "  30/11/2011 17:12   0   0   0   0   0"                                           
##  [32] ""                                                                                 
##  [33] ", , user_name = carlitos"                                                         
##  [34] ""                                                                                 
##  [35] "                  classe"                                                         
##  [36] "cvtd_timestamp       A   B   C   D   E"                                           
##  [37] "  02/12/2011 13:32   0   0   0   0   0"                                           
##  [38] "  02/12/2011 13:33   0   0   0   0   0"                                           
##  [39] "  02/12/2011 13:34   0   0   0   0   0"                                           
##  [40] "  02/12/2011 13:35   0   0   0   0   0"                                           
##  [41] "  02/12/2011 14:56   0   0   0   0   0"                                           
##  [42] "  02/12/2011 14:57   0   0   0   0   0"                                           
##  [43] "  02/12/2011 14:58   0   0   0   0   0"                                           
##  [44] "  02/12/2011 14:59   0   0   0   0   0"                                           
##  [45] "  05/12/2011 11:23 190   0   0   0   0"                                           
##  [46] "  05/12/2011 11:24 644 690 163   0   0"                                           
##  [47] "  05/12/2011 11:25   0   0 330 486 609"                                           
##  [48] "  05/12/2011 14:22   0   0   0   0   0"                                           
##  [49] "  05/12/2011 14:23   0   0   0   0   0"                                           
##  [50] "  05/12/2011 14:24   0   0   0   0   0"                                           
##  [51] "  28/11/2011 14:13   0   0   0   0   0"                                           
##  [52] "  28/11/2011 14:14   0   0   0   0   0"                                           
##  [53] "  28/11/2011 14:15   0   0   0   0   0"                                           
##  [54] "  30/11/2011 17:10   0   0   0   0   0"                                           
##  [55] "  30/11/2011 17:11   0   0   0   0   0"                                           
##  [56] "  30/11/2011 17:12   0   0   0   0   0"                                           
##  [57] ""                                                                                 
##  [58] ", , user_name = charles"                                                          
##  [59] ""                                                                                 
##  [60] "                  classe"                                                         
##  [61] "cvtd_timestamp       A   B   C   D   E"                                           
##  [62] "  02/12/2011 13:32   0   0   0   0   0"                                           
##  [63] "  02/12/2011 13:33   0   0   0   0   0"                                           
##  [64] "  02/12/2011 13:34   0   0   0   0   0"                                           
##  [65] "  02/12/2011 13:35   0   0   0   0   0"                                           
##  [66] "  02/12/2011 14:56 235   0   0   0   0"                                           
##  [67] "  02/12/2011 14:57 664 716   0   0   0"                                           
##  [68] "  02/12/2011 14:58   0  29 539 642 154"                                           
##  [69] "  02/12/2011 14:59   0   0   0   0 557"                                           
##  [70] "  05/12/2011 11:23   0   0   0   0   0"                                           
##  [71] "  05/12/2011 11:24   0   0   0   0   0"                                           
##  [72] "  05/12/2011 11:25   0   0   0   0   0"                                           
##  [73] "  05/12/2011 14:22   0   0   0   0   0"                                           
##  [74] "  05/12/2011 14:23   0   0   0   0   0"                                           
##  [75] "  05/12/2011 14:24   0   0   0   0   0"                                           
##  [76] "  28/11/2011 14:13   0   0   0   0   0"                                           
##  [77] "  28/11/2011 14:14   0   0   0   0   0"                                           
##  [78] "  28/11/2011 14:15   0   0   0   0   0"                                           
##  [79] "  30/11/2011 17:10   0   0   0   0   0"                                           
##  [80] "  30/11/2011 17:11   0   0   0   0   0"                                           
##  [81] "  30/11/2011 17:12   0   0   0   0   0"                                           
##  [82] ""                                                                                 
##  [83] ", , user_name = eurico"                                                           
##  [84] ""                                                                                 
##  [85] "                  classe"                                                         
##  [86] "cvtd_timestamp       A   B   C   D   E"                                           
##  [87] "  02/12/2011 13:32   0   0   0   0   0"                                           
##  [88] "  02/12/2011 13:33   0   0   0   0   0"                                           
##  [89] "  02/12/2011 13:34   0   0   0   0   0"                                           
##  [90] "  02/12/2011 13:35   0   0   0   0   0"                                           
##  [91] "  02/12/2011 14:56   0   0   0   0   0"                                           
##  [92] "  02/12/2011 14:57   0   0   0   0   0"                                           
##  [93] "  02/12/2011 14:58   0   0   0   0   0"                                           
##  [94] "  02/12/2011 14:59   0   0   0   0   0"                                           
##  [95] "  05/12/2011 11:23   0   0   0   0   0"                                           
##  [96] "  05/12/2011 11:24   0   0   0   0   0"                                           
##  [97] "  05/12/2011 11:25   0   0   0   0   0"                                           
##  [98] "  05/12/2011 14:22   0   0   0   0   0"                                           
##  [99] "  05/12/2011 14:23   0   0   0   0   0"                                           
## [100] "  05/12/2011 14:24   0   0   0   0   0"                                           
## [101] "  28/11/2011 14:13 833   0   0   0   0"                                           
## [102] "  28/11/2011 14:14  32 592 489 385   0"                                           
## [103] "  28/11/2011 14:15   0   0   0 197 542"                                           
## [104] "  30/11/2011 17:10   0   0   0   0   0"                                           
## [105] "  30/11/2011 17:11   0   0   0   0   0"                                           
## [106] "  30/11/2011 17:12   0   0   0   0   0"                                           
## [107] ""                                                                                 
## [108] ", , user_name = jeremy"                                                           
## [109] ""                                                                                 
## [110] "                  classe"                                                         
## [111] "cvtd_timestamp       A   B   C   D   E"                                           
## [112] "  02/12/2011 13:32   0   0   0   0   0"                                           
## [113] "  02/12/2011 13:33   0   0   0   0   0"                                           
## [114] "  02/12/2011 13:34   0   0   0   0   0"                                           
## [115] "  02/12/2011 13:35   0   0   0   0   0"                                           
## [116] "  02/12/2011 14:56   0   0   0   0   0"                                           
## [117] "  02/12/2011 14:57   0   0   0   0   0"                                           
## [118] "  02/12/2011 14:58   0   0   0   0   0"                                           
## [119] "  02/12/2011 14:59   0   0   0   0   0"                                           
## [120] "  05/12/2011 11:23   0   0   0   0   0"                                           
## [121] "  05/12/2011 11:24   0   0   0   0   0"                                           
## [122] "  05/12/2011 11:25   0   0   0   0   0"                                           
## [123] "  05/12/2011 14:22   0   0   0   0   0"                                           
## [124] "  05/12/2011 14:23   0   0   0   0   0"                                           
## [125] "  05/12/2011 14:24   0   0   0   0   0"                                           
## [126] "  28/11/2011 14:13   0   0   0   0   0"                                           
## [127] "  28/11/2011 14:14   0   0   0   0   0"                                           
## [128] "  28/11/2011 14:15   0   0   0   0   0"                                           
## [129] "  30/11/2011 17:10 869   0   0   0   0"                                           
## [130] "  30/11/2011 17:11 308 489 643   0   0"                                           
## [131] "  30/11/2011 17:12   0   0   9 522 562"                                           
## [132] ""                                                                                 
## [133] ", , user_name = pedro"                                                            
## [134] ""                                                                                 
## [135] "                  classe"                                                         
## [136] "cvtd_timestamp       A   B   C   D   E"                                           
## [137] "  02/12/2011 13:32   0   0   0   0   0"                                           
## [138] "  02/12/2011 13:33   0   0   0   0   0"                                           
## [139] "  02/12/2011 13:34   0   0   0   0   0"                                           
## [140] "  02/12/2011 13:35   0   0   0   0   0"                                           
## [141] "  02/12/2011 14:56   0   0   0   0   0"                                           
## [142] "  02/12/2011 14:57   0   0   0   0   0"                                           
## [143] "  02/12/2011 14:58   0   0   0   0   0"                                           
## [144] "  02/12/2011 14:59   0   0   0   0   0"                                           
## [145] "  05/12/2011 11:23   0   0   0   0   0"                                           
## [146] "  05/12/2011 11:24   0   0   0   0   0"                                           
## [147] "  05/12/2011 11:25   0   0   0   0   0"                                           
## [148] "  05/12/2011 14:22 267   0   0   0   0"                                           
## [149] "  05/12/2011 14:23 373 505 492   0   0"                                           
## [150] "  05/12/2011 14:24   0   0   7 469 497"                                           
## [151] "  28/11/2011 14:13   0   0   0   0   0"                                           
## [152] "  28/11/2011 14:14   0   0   0   0   0"                                           
## [153] "  28/11/2011 14:15   0   0   0   0   0"                                           
## [154] "  30/11/2011 17:10   0   0   0   0   0"                                           
## [155] "  30/11/2011 17:11   0   0   0   0   0"                                           
## [156] "  30/11/2011 17:12   0   0   0   0   0"                                           
## [157] ""                                                                                 
## [158] " "                                                                                
## [159] ""                                                                                 
## [160] ""
```

```r
close(con)
```


<br>

<br>
<h4 id="App2">Appendix 2: Log of Random Forest Model Data</h4>


```r
con <- file("Documentation/Random_Forest_Model.txt", "r", blocking = FALSE)
readLines("Documentation/Random_Forest_Model.txt")
```

```
##   [1] "##################################################### "                
##   [2] "# This is a Log Capturing the Random Forest Tree Results"              
##   [3] "# Created:  Mon Jan  2 08:17:40 2017 "                                 
##   [4] "##################################################### "                
##   [5] " "                                                                     
##   [6] " "                                                                     
##   [7] ""                                                                      
##   [8] "                             A         B         C         D         E"
##   [9] "num_window           16.024718 20.763105 22.884317 21.042931 18.135973"
##  [10] "roll_belt            15.061804 20.069067 18.121956 21.421957 17.159104"
##  [11] "pitch_belt           11.521232 20.271608 16.186241 18.682291 16.258959"
##  [12] "yaw_belt             17.761668 21.710618 21.203173 21.762556 14.281758"
##  [13] "total_accel_belt      7.985035 10.013000  8.026394  8.324065  7.294517"
##  [14] "gyros_belt_x          6.489707  9.222908  8.962467  6.754466 11.958591"
##  [15] "gyros_belt_y          5.062034  7.525845  6.663050  6.655441  9.447508"
##  [16] "gyros_belt_z          9.860442 12.899770 10.312663 10.699562 12.141820"
##  [17] "accel_belt_x          6.723981  9.113073  8.819301  7.318663  7.538151"
##  [18] "accel_belt_y          6.580888  6.292784  7.093961  7.721691  5.768616"
##  [19] "accel_belt_z          8.929455  9.790304 10.851120 10.403680  9.073536"
##  [20] "magnet_belt_x         7.643335 15.024559 10.940130 11.507913 11.324437"
##  [21] "magnet_belt_y        10.164274 11.661418 11.841930 10.460043  9.930806"
##  [22] "magnet_belt_z         8.610183 11.757244 10.537555 13.671386 11.347376"
##  [23] "roll_arm              9.633819 11.718844 11.389241 12.418387  9.689523"
##  [24] "pitch_arm             7.073879 10.688642  9.337041  8.702393  8.319898"
##  [25] "yaw_arm               9.173545 11.939660 10.645555 11.988869  9.033401"
##  [26] "total_accel_arm       4.065773 12.255909  9.802004  8.862552  9.268660"
##  [27] "gyros_arm_x           6.837997 11.437193 10.861925 10.785481  9.099866"
##  [28] "gyros_arm_y           7.971706 14.940366 14.673269 12.732876 14.524770"
##  [29] "gyros_arm_z           5.875256  8.059275  8.957054  6.546702  5.483843"
##  [30] "accel_arm_x           8.073746  9.365019  9.578887 11.111666  9.078450"
##  [31] "accel_arm_y           8.930798  9.947869  8.872118  8.288839  8.252510"
##  [32] "accel_arm_z           5.751341 11.962215 10.149486 11.749863  9.531502"
##  [33] "magnet_arm_x          6.816815  6.693087  7.484870  7.446436  6.181297"
##  [34] "magnet_arm_y          5.344110  7.884725  8.305119  9.261945  4.847302"
##  [35] "magnet_arm_z          8.965879  9.517644 10.359628  8.870976  7.686243"
##  [36] "roll_dumbbell         9.505241 11.000897 11.837891 11.557742 11.799073"
##  [37] "pitch_dumbbell        5.788875 10.194994  6.994982  6.316711  6.727654"
##  [38] "yaw_dumbbell          7.643916 10.647247 11.281808  9.802804  9.502272"
##  [39] "total_accel_dumbbell 10.199545 10.755427 10.441957 10.456604 11.596779"
##  [40] "gyros_dumbbell_x      6.489330 12.204698 10.340114 10.152866 10.660845"
##  [41] "gyros_dumbbell_y      9.501006 11.032729 12.581875  9.683094  8.360437"
##  [42] "gyros_dumbbell_z      9.068930 10.599558 10.667101  9.234897  8.032108"
##  [43] "accel_dumbbell_x      6.808127 10.951642  8.703480  8.802517 12.044264"
##  [44] "accel_dumbbell_y     13.243606 15.250873 12.722820 13.034037 12.410119"
##  [45] "accel_dumbbell_z      9.749849 10.720788 10.541446 10.879988 13.819947"
##  [46] "magnet_dumbbell_x    11.123105 11.951287 12.775915 12.074963 11.065135"
##  [47] "magnet_dumbbell_y    16.584411 17.133598 18.400051 15.858149 15.173647"
##  [48] "magnet_dumbbell_z    19.610103 19.604542 20.571914 18.842012 17.182463"
##  [49] "roll_forearm         12.012509 10.960342 12.472810 10.011947  9.625897"
##  [50] "pitch_forearm        15.495576 16.432206 18.601045 14.862421 14.638193"
##  [51] "yaw_forearm           9.091583  8.621935  9.578849  8.638759  8.376614"
##  [52] "total_accel_forearm   7.476333  9.409764  9.291175  9.505572  9.274844"
##  [53] "gyros_forearm_x       6.171802  9.217549  8.574533  7.464346  7.878066"
##  [54] "gyros_forearm_y       6.029547 12.686295 12.245724 11.465795  8.036672"
##  [55] "gyros_forearm_z       7.281156 10.690183  9.885007 11.040826  9.143009"
##  [56] "accel_forearm_x       7.013529 12.469541 11.209159 13.021706 10.339376"
##  [57] "accel_forearm_y       6.755595 10.711374 10.587576  9.340770 10.125554"
##  [58] "accel_forearm_z       9.568225 10.744983 10.790727 10.122767 11.917588"
##  [59] "magnet_forearm_x      6.253818  9.352312  8.039123  8.192869  8.275685"
##  [60] "magnet_forearm_y      8.626153  9.361469  8.495428  7.425499  7.871474"
##  [61] "magnet_forearm_z      8.811970 14.265156 13.322005 14.713244 11.355168"
##  [62] "                     MeanDecreaseAccuracy MeanDecreaseGini"            
##  [63] "num_window                      21.812618       2086.02468"            
##  [64] "roll_belt                       21.637675       1627.72503"            
##  [65] "pitch_belt                      20.668116        959.11980"            
##  [66] "yaw_belt                        27.520206       1133.04870"            
##  [67] "total_accel_belt                10.257521        325.90343"            
##  [68] "gyros_belt_x                    10.374099        122.18924"            
##  [69] "gyros_belt_y                     7.983096        158.39258"            
##  [70] "gyros_belt_z                    13.564700        375.72891"            
##  [71] "accel_belt_x                    11.340411        157.84164"            
##  [72] "accel_belt_y                     8.568848        169.02132"            
##  [73] "accel_belt_z                    12.143365        501.03548"            
##  [74] "magnet_belt_x                   14.350032        342.96042"            
##  [75] "magnet_belt_y                   12.517295        468.36757"            
##  [76] "magnet_belt_z                   12.906328        515.44606"            
##  [77] "roll_arm                        13.790242        424.43246"            
##  [78] "pitch_arm                       10.859147        230.57481"            
##  [79] "yaw_arm                         12.943445        295.89065"            
##  [80] "total_accel_arm                 10.170574        117.55136"            
##  [81] "gyros_arm_x                     11.000457        163.76744"            
##  [82] "gyros_arm_y                     18.088131        152.42418"            
##  [83] "gyros_arm_z                     11.060543         63.26546"            
##  [84] "accel_arm_x                     10.295814        325.42604"            
##  [85] "accel_arm_y                     11.223355        180.45479"            
##  [86] "accel_arm_z                     12.350177        136.62487"            
##  [87] "magnet_arm_x                     7.222631        315.26480"            
##  [88] "magnet_arm_y                     7.480713        249.11900"            
##  [89] "magnet_arm_z                    10.912352        194.22427"            
##  [90] "roll_dumbbell                   12.252131        546.27555"            
##  [91] "pitch_dumbbell                   7.869416        231.89803"            
##  [92] "yaw_dumbbell                    12.189236        297.90259"            
##  [93] "total_accel_dumbbell            12.833809        341.31610"            
##  [94] "gyros_dumbbell_x                15.813607        139.85569"            
##  [95] "gyros_dumbbell_y                12.207088        281.77080"            
##  [96] "gyros_dumbbell_z                14.504690         90.19071"            
##  [97] "accel_dumbbell_x                10.305355        381.66965"            
##  [98] "accel_dumbbell_y                16.207887        511.56615"            
##  [99] "accel_dumbbell_z                13.250517        448.37199"            
## [100] "magnet_dumbbell_x               12.976302        613.45545"            
## [101] "magnet_dumbbell_y               19.609452        900.27515"            
## [102] "magnet_dumbbell_z               24.019058       1012.00935"            
## [103] "roll_forearm                    11.830875        735.83625"            
## [104] "pitch_forearm                   19.469173       1046.87957"            
## [105] "yaw_forearm                     10.995771        197.48178"            
## [106] "total_accel_forearm             11.256279        122.83480"            
## [107] "gyros_forearm_x                 11.727381         85.62149"            
## [108] "gyros_forearm_y                 14.257345        130.06487"            
## [109] "gyros_forearm_z                 12.544092        100.66151"            
## [110] "accel_forearm_x                 11.552417        409.22500"            
## [111] "accel_forearm_y                 11.502167        148.49568"            
## [112] "accel_forearm_z                 13.374053        302.88610"            
## [113] "magnet_forearm_x                 8.652204        249.72640"            
## [114] "magnet_forearm_y                 9.600141        254.96171"            
## [115] "magnet_forearm_z                17.073383        345.73828"            
## [116] " "                                                                     
## [117] ""                                                                      
## [118] " "                                                                     
## [119] ""                                                                      
## [120] ""                                                                      
## [121] " "                                                                     
## [122] ""                                                                      
## [123] ""
```

```r
close(con)
```

<br>
<h4 id="App3">Appendix 3: Log of Simplified Random Forest Model Data</h4>


```r
con <- file("Documentation/Random_Forest_Model2.txt", "r", blocking = FALSE)
readLines("Documentation/Random_Forest_Model2.txt")
```

```
##  [1] "##################################################### "             
##  [2] "# This is a Log Capturing the Simplified Random Forest Tree Results"
##  [3] "# Created:  Mon Jan  2 08:25:33 2017 "                              
##  [4] "##################################################### "             
##  [5] " "                                                                  
##  [6] " "                                                                  
##  [7] ""                                                                   
##  [8] "Confusion Matrix and Statistics"                                    
##  [9] ""                                                                   
## [10] "          Reference"                                                
## [11] "Prediction   A   B   C   D   E"                                     
## [12] "         A 499   0   0   0   0"                                     
## [13] "         B   0 347   1   0   0"                                     
## [14] "         C   0   0 316   1   0"                                     
## [15] "         D   0   0   0 299   0"                                     
## [16] "         E   0   0   0   0 344"                                     
## [17] ""                                                                   
## [18] "Overall Statistics"                                                 
## [19] "                                         "                          
## [20] "               Accuracy : 0.9989         "                          
## [21] "                 95% CI : (0.996, 0.9999)"                          
## [22] "    No Information Rate : 0.2761         "                          
## [23] "    P-Value [Acc > NIR] : < 2.2e-16      "                          
## [24] "                                         "                          
## [25] "                  Kappa : 0.9986         "                          
## [26] " Mcnemar's Test P-Value : NA             "                          
## [27] ""                                                                   
## [28] "Statistics by Class:"                                               
## [29] ""                                                                   
## [30] "                     Class: A Class: B Class: C Class: D Class: E"  
## [31] "Sensitivity            1.0000   1.0000   0.9968   0.9967   1.0000"  
## [32] "Specificity            1.0000   0.9993   0.9993   1.0000   1.0000"  
## [33] "Pos Pred Value         1.0000   0.9971   0.9968   1.0000   1.0000"  
## [34] "Neg Pred Value         1.0000   1.0000   0.9993   0.9993   1.0000"  
## [35] "Prevalence             0.2761   0.1920   0.1754   0.1660   0.1904"  
## [36] "Detection Rate         0.2761   0.1920   0.1749   0.1655   0.1904"  
## [37] "Detection Prevalence   0.2761   0.1926   0.1754   0.1655   0.1904"  
## [38] "Balanced Accuracy      1.0000   0.9997   0.9981   0.9983   1.0000"  
## [39] " "                                                                  
## [40] ""                                                                   
## [41] " "                                                                  
## [42] ""                                                                   
## [43] "                         A         B         C         D         E" 
## [44] "num_window       31.361124 37.561341 44.682921 32.721379 21.793081" 
## [45] "roll_belt        17.219486 18.678343 21.290510 19.924290 17.445971" 
## [46] "pitch_belt       20.887441 20.632304 18.189963 17.307170 16.336910" 
## [47] "yaw_belt         16.281578 21.135192 17.641623 27.270936 15.397867" 
## [48] "total_accel_belt  8.137787  7.815665  6.274316  8.234807  7.489202" 
## [49] "gyros_belt_x     12.311223 10.787975 14.091696 10.619257 13.924750" 
## [50] "gyros_belt_y      6.860335  8.282067  8.535046  8.610661  9.379573" 
## [51] "gyros_belt_z     12.242936 10.992555 12.326802 10.812217 10.498117" 
## [52] "accel_belt_x     10.673199 10.487307 12.052757  8.749628 12.620318" 
## [53] "accel_belt_y      7.818817  7.701176  7.169086  8.541795  6.048712" 
## [54] "accel_belt_z     11.259712  9.816876 13.752961 12.496770 11.501801" 
## [55] "magnet_belt_x    11.070942 12.467056 14.485769 12.519904 14.898076" 
## [56] "magnet_belt_y    12.102507 10.831155 12.788134 15.003102 13.038608" 
## [57] "magnet_belt_z    13.911470 15.311670 16.837849 18.751685 12.188533" 
## [58] "roll_arm         18.463348 15.898673 15.308792 19.813159 15.911326" 
## [59] "                 MeanDecreaseAccuracy MeanDecreaseGini"             
## [60] "num_window                  38.953899        6308.4931"             
## [61] "roll_belt                   22.907072        2756.5043"             
## [62] "pitch_belt                  23.240817        2112.7697"             
## [63] "yaw_belt                    24.488686        2473.3230"             
## [64] "total_accel_belt             9.201700         375.1985"             
## [65] "gyros_belt_x                17.694883         365.8504"             
## [66] "gyros_belt_y                10.931989         267.9115"             
## [67] "gyros_belt_z                13.472726         643.4181"             
## [68] "accel_belt_x                13.483021         539.0120"             
## [69] "accel_belt_y                 8.942374         396.6869"             
## [70] "accel_belt_z                13.737058        1025.0580"             
## [71] "magnet_belt_x               13.941318        1065.3088"             
## [72] "magnet_belt_y               14.615295         892.6668"             
## [73] "magnet_belt_z               19.179077        1024.3218"             
## [74] "roll_arm                    22.106129        1474.0091"             
## [75] " "                                                                  
## [76] ""                                                                   
## [77] " "                                                                  
## [78] ""                                                                   
## [79] ""                                                                   
## [80] " "                                                                  
## [81] ""                                                                   
## [82] ""
```

```r
close(con)
```

<br>
<h4 id="SysVers">System and Version Infomation</h4>



| Code Origininally Executed on: | Value                 |
| ------------------------------ |----------------------:|
| R Version                      | R version 3.3.2 (2016-10-31)  |
| Operating System               | darwin13.4.0              |
| Architecture                   | x86_64            |

<br>Return to <a href="#Top">top</a>.
<p align="right">File created *Mon Jan  2 08:25:35 2017*</p> 
