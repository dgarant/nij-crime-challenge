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

models <- fromJSON("~/repos/nij-crime-challenge/models/poisson/models.json")
split.test <- split(test.d, test.d$groupid)

group.forecasts <- list()
write.header <- TRUE
for(groupid in names(split.test)) {
  all.group.test <- split.test[[groupid]]
  for(outcome in names(outcome.vars)) {
    group.model <- models[[groupid]][[outcome.vars[[outcome]]]]
    if(group.model$model_type == "median") {
      preds <- rep(group.model$mean_value, nrow(all.group.test))
    } else if(group.model$model_type == "poisson") {
       predictor.cols <- predictors[1:length(group.model$coeffs)]
       preds <- exp(as.matrix(all.group.test[, predictor.cols]) %*% group.model$coeffs)[, 1]
    } else {
      stop(paste0("Invalid model type ", group.model$model_type, "\n"))
    }
    
    all.group.test[, paste0("pred_", outcome)] <- preds
  }
  group.forecasts[[groupid]] <- all.group.test[, c("cell_id", "forecast_start", paste0("pred_", names(outcome.vars)), names(outcome.vars))]
  
  write.table(group.forecasts[[groupid]], "~/repos/nij-crime-challenge/models/poisson/forecast.csv", 
            col.names=write.header, append=!write.header, row.names=FALSE, quote=FALSE, sep=",")
  write.header <- FALSE
}

global.models <- fromJSON("~/repos/nij-crime-challenge/models/poisson/initial-weights.json")
test.result <- data.frame(
  cell_id = test.d$cell_id,
  forecast_start = test.d$forecast_start)
for(outcome in names(outcome.vars)) {
  preds <- exp(as.matrix(test.d[, predictors]) %*% global.models[[outcome]])[, 1]
  test.result[, outcome] <- test.d[, outcome]
  test.result[, paste0("pred_", outcome)] <- preds
}
write.table(test.result, "~/repos/nij-crime-challenge/models/poisson/forecast-global.csv", 
            row.names=FALSE, quote=FALSE, sep=",", col.names=TRUE)
