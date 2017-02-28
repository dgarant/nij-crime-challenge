options(warn=1)
library(stringr)
library(plyr)
library(kernlab)
library(jsonlite)
library(data.table)

cat("Reading data\n")
d <- fread("gzcat ~/repos/nij-crime-challenge/features/processed-features-550.csv.gz", data.table=FALSE, check.names=TRUE)

outcome.vars <- list()
for(window in c(7, 14, 30, 61, 91)) {
  for(type in c("BURGLARY", "ALL", "STREET CRIMES", "MOTOR VEHICLE THEFT")) {
    name <- paste0("outcome_num_crimes_", window, "days_", str_replace_all(type, " ", "."))
    dispname <- paste0("outcome_num_crimes_", window, "days_", type)
    outcome.vars[[name]] <- dispname
  }
}

predictors <- colnames(d)[str_detect(colnames(d), "p_")]

train.d <- subset(d, istrain == 1)
test.d <- subset(d, istrain == 0)

weights.path <- "~/repos/nij-crime-challenge/models/poisson/initial-weights.json"
if(!file.exists(weights.path)) {
  init.weights <- list()
  for(outcome in names(outcome.vars)) {
    cat("Fitting global model of ", outcome, "\n")
    outcome.formula <- as.formula(paste0(outcome, "~", paste0(predictors, collapse=" + "), " - 1"))
    init.weights[[outcome]] <- coefficients(glm(outcome.formula, data=train.d, family="poisson"))
  }
  
  conn <- file(weights.path)
  jweights <- toJSON(init.weights, pretty=TRUE)
  writeLines(jweights, conn)
  close(conn)
} else {
  init.weights <- fromJSON(weights.path)
}

split.train <- split(train.d, train.d$groupid)
split.test <- split(test.d, test.d$groupid)
outcome.models <- list()
for(grecs in split.train) {
  
  groupid <- as.character(grecs$groupid[1])
  test.recs <- split.test[[groupid]]
  outcome.models[[groupid]] <- list()
  cat("Working on group", groupid, "\n")
  
  for(outcome in names(outcome.vars)) {
    distr.stats <- quantile(grecs[, outcome], probs=c(0.5, 0.95))
    mmodel <- list(model_type = unbox("median"), "median_value" = unbox(distr.stats["50%"]), 
                   "mean_value" = unbox(mean(grecs[, outcome])), "domain"=unique(grecs[, outcome]))
    if(distr.stats["95%"] < 1) {
      outcome.models[[groupid]][[outcome.vars[[outcome]]]] = mmodel
      next
    } else if(sd(grecs[, outcome]) < 1) {
      cur.predictors <- predictors[1:7]
    } else if(sd(grecs[, outcome]) < 4) {
      cur.predictors <- predictors[1:15]
    } else {
      cur.predictors <- predictors[1:50]
    }
    outcome.formula <- as.formula(paste0(outcome, "~", paste0(cur.predictors, collapse=" + "), " - 1"))
    
    omodel <- tryCatch({
      pmodel <- glm(outcome.formula, data=grecs, family="poisson")
      pred.counts <- predict(pmodel, test.recs, type="response")
      rmse <- sqrt(mean((pred.counts - test.recs[, outcome]))** 2)
      nrmse <- rmse / mean(grecs[, outcome])
      
      bw.estimates <- as.list(1/sigest(as.matrix(grecs[, outcome]), scale=FALSE))
      list(model_type = unbox("poisson"), 
                  "median_value" = unbox(distr.stats["50%"]), 
                  "mean_value" = unbox(mean(grecs[, outcome])), 
                  domain = unique(grecs[, outcome]),
                  "coeffs" = coefficients(pmodel),
                  "bw" = bw.estimates,
                  "rmse" = unbox(rmse),
                  "nrmse" = unbox(nrmse))
    }, error=function(e) {
      print(e)
      cat("\tCould not fit model to ", outcome, ", using mean\n")
      return(mmodel)
    }, warning=function(e) {
      print(e)
      if(sum(grecs[, outcome]) > 100 && str_detect(e$message, "converge")) {
        stop("convergence problem with lots of non-zero cases")
      }
      return(list(model_type = unbox("poisson"), 
                  "median_value" = unbox(distr.stats["50%"]), 
                  "mean_value" = unbox(mean(grecs[, outcome])), 
                  domain = unique(grecs[, outcome]),
                  "coeffs" = coefficients(pmodel),
                  "bw" = bw.estimates,
                  "rmse" = unbox(rmse),
                  "nrmse" = unbox(nrmse)))
    })
    
    outcome.models[[groupid]][[outcome.vars[[outcome]]]] <- omodel
  }
}

conn <- file("~/repos/nij-crime-challenge/models/poisson/models.json")
model.string <- toJSON(outcome.models, pretty=TRUE)
writeLines(model.string, conn)
close(conn)
