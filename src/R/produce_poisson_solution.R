options(warn=1)
library(stringr)
library(plyr)
library(ggplot2)
library(jsonlite)
library(data.table)
global.forecast <- fread("~/repos/nij-crime-challenge/models/poisson/forecast-global.csv", data.table=FALSE)
indiv.forecast <- fread("~/repos/nij-crime-challenge/models/poisson/forecast.csv", data.table=FALSE)

predictors <- colnames(global.forecast)[str_detect(colnames(global.forecast), "pred_")]
pforecast <- merge(indiv.forecast, global.forecast[, c("cell_id", "forecast_start", predictors)], 
                   suffixes=c("", ".global"), by=c("cell_id", "forecast_start"))

cell.meta <- fread("~/repos/nij-crime-challenge/models/cells/cells-dim-550-meta.csv", data.table=FALSE)

outcome.vars <- list()
for(window in c(7, 14, 30, 61, 91)) {
  for(type in c("BURGLARY", "ALL", "STREET CRIMES", "MOTOR VEHICLE THEFT")) {
    name <- paste0("outcome_num_crimes_", window, "days_", str_replace_all(type, " ", "."))
    dispname <- paste0("outcome_num_crimes_", window, "days_", type)
    outcome.vars[[name]] <- dispname
  }
}

#examine.dates <- c("2013-03-15", "2015-09-15", "2016-04-15")
examine.dates <- unique(subset(pforecast, forecast_start < "2017-12-15")$forecast_start)

eval.pei <- function(forecast, outcome.var) {
  # forecast at least 25 cells, not more than 60
  pred.outcome <- paste0("pred_", outcome.var)
  global.pred.outcome <- paste0(pred.outcome, ".global")
  if(str_detect(outcome.var, "ALL")) {
    baseline.outcome <- "num.crimes"
  } else {
    type <- tail(str_split(outcome.var, "_")[[1]], n=1)
    baseline.outcome <- paste0("num.crimes.", type)
  }
  
  ordered.by.preds <- forecast[order(-forecast[, pred.outcome]), ]
  ordered.by.global.preds <- forecast[order(-forecast[, global.pred.outcome]), ]
  ordered.by.actuals <- forecast[order(-forecast[, outcome.var]), ]
  ordered.by.baseline <- forecast[order(-forecast[, baseline.outcome]), ]
  
  #print(ordered.by.actuals[1:100, c("cell_id", "forecast_start", pred.outcome, outcome.var, "num.crimes")])
  #print(ordered.by.baseline[1:100, c("cell_id", "forecast_start", pred.outcome, outcome.var, "num.crimes")])
  #cat("ordered by preds:\n")
  #print(ordered.by.preds[1:100, c("cell_id", "forecast_start", pred.outcome, outcome.var, "num.crimes")])
  
  pei.by.ncells <- NULL
  for(num.cells in seq(25, 60)) {
    forecast.num <- sum(ordered.by.preds[1:num.cells, outcome.var])
    global.num <- sum(ordered.by.global.preds[1:num.cells, outcome.var])
    baseline.num <- sum(ordered.by.baseline[1:num.cells, outcome.var])
    actual.num <- sum(ordered.by.actuals[1:num.cells, outcome.var])
    
    pei <- forecast.num / actual.num
    pei.baseline <- baseline.num / actual.num
    pei.global <- global.num / actual.num
    
    pei.by.ncells <- rbind(pei.by.ncells, data.frame(date=forecast$forecast_start[1], 
                                                     cells=num.cells, pei=pei, 
                                                     pei.baseline=pei.baseline, pei.global=pei.global))
  }
  return(pei.by.ncells)
}

perf.summary <- NULL
for(outcome.var in names(outcome.vars)) {
  all.pei.by.cells <- NULL
  for(d in examine.dates) {
    pred.outcome.var <- paste0("pred_", outcome.var)
    
    date.records <- subset(pforecast, forecast_start == d)
    
    cell.outcomes <- merge(date.records, cell.meta, by.x="cell_id", by.y="id")
    
    pei.by.cells <- eval.pei(cell.outcomes, outcome.var)
    all.pei.by.cells <- rbind(all.pei.by.cells, pei.by.cells)
  }
  
  pei.melted <- melt(all.pei.by.cells, id.vars=c("date", "cells"))
  g <- ggplot(pei.melted, aes(x=cells, y=value, fill=variable, group=interaction(cells, variable))) + 
    geom_boxplot(outlier.size=0) + labs(title=outcome.var) + geom_jitter(aes(color=variable), alpha=0.2)
    #geom_text(data=subset(pei.melted, value < 0.5), aes(label=date), size=3) + 
  print(g)
  
  med.peis <- ddply(pei.melted, .(variable, cells), summarize, med.pei=median(value))
  best.outcome.cells <- ddply(med.peis, .(variable), function(df) {
    best.cells <- df$cells[which.max(df$med.pei)]
    return(data.frame(best.cells=best.cells, best.pei=max(df$med.pei)))
  })
  
  perf.summary <- rbind(perf.summary, best.outcome.cells)
}
print(perf.summary)


output.date <- "2017-03-01"

# TODO: generate solution directories
output.dirs <- c("Burg"="BURGLARY", "ACFS"="ALL", "SC"="STREET.CRIMES", "TOA"="MOTOR.VEHICLE.THEFT")
output.windows <- c("1MO"=7, "1WK"=14, "2MO"=30, "2WK"=61, "3MO"=91)
base.dir <- "~/repos/nij-crime-challenge/models/poisson/GARANT_ANALYTICS"
for(type in names(output.dirs)) {
  for(window in names(output.windows)) {
    dir.create(paste0(base.dir, "/", type, "/", window), recursive=TRUE, showWarnings=FALSE)
  }
}
