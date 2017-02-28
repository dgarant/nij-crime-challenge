options(warn=1)
library(stringr)
library(plyr)
library(ggplot2)
library(jsonlite)
library(data.table)

library(maptools)
library(rgdal)
library(rgeos)

global.forecast <- fread("~/repos/nij-crime-challenge/models/poisson/forecast-global.csv", data.table=FALSE)
indiv.forecast <- fread("~/repos/nij-crime-challenge/models/poisson/forecast.csv", data.table=FALSE)

predictors <- colnames(global.forecast)[str_detect(colnames(global.forecast), "pred_")]
pforecast <- merge(indiv.forecast, global.forecast[, c("cell_id", "forecast_start", predictors)], 
                   suffixes=c("", ".global"), by=c("cell_id", "forecast_start"))

cell.meta <- fread("~/repos/nij-crime-challenge/models/cells/cells-dim-550-meta.csv", data.table=FALSE)

cell.outcomes <- merge(pforecast, cell.meta, by.x="cell_id", by.y="id")

outcome.vars <- list()
for(window in c(7, 14, 30, 61, 91)) {
  for(type in c("BURGLARY", "ALL", "STREET CRIMES", "MOTOR VEHICLE THEFT")) {
    name <- paste0("outcome_num_crimes_", window, "days_", str_replace_all(type, " ", "."))
    dispname <- paste0("outcome_num_crimes_", window, "days_", type)
    outcome.vars[[name]] <- dispname
  }
}

#examine.dates <- c("2013-03-15", "2015-09-15", "2016-04-15")
examine.dates <- unique(subset(pforecast, forecast_start < "2016-11-15")$forecast_start)

get.ranked.cells <- function(forecast, model.type, outcome.var) {
  if(model.type == "baseline") {
    if(str_detect(outcome.var, "ALL")) {
      baseline.outcome <- "num.crimes"
    } else {
      type <- tail(str_split(outcome.var, "_")[[1]], n=1)
      baseline.outcome <- paste0("num.crimes.", type)
    }
    
    pred.var <- baseline.outcome
  } else if(model.type == "global") {
    pred.var <- paste0("pred_", outcome.var, ".global")
  } else if(model.type == "indiv") {
    pred.var <- paste0("pred_", outcome.var)
  } else if(model.type == "actual") {
    pred.var <- outcome.var
  } else {
    stop(paste0("unknown model type ", model.type))
  }
  
  ranked <- forecast[order(-forecast[, pred.var]), ]
  return(data.frame(cell_id=ranked$cell_id, pred=ranked[, pred.var], actual=ranked[, outcome.var]))
}

eval.pei <- function(forecast, outcome.var) {
  # forecast at least 25 cells, not more than 60
  
  ordered.by.preds <- get.ranked.cells(forecast, "indiv", outcome.var)
  ordered.by.global.preds <- get.ranked.cells(forecast, "global", outcome.var)
  ordered.by.actuals <- get.ranked.cells(forecast, "actual", outcome.var)
  ordered.by.baseline <- get.ranked.cells(forecast, "baseline", outcome.var)
  
  #print(ordered.by.actuals[1:100, c("cell_id", "forecast_start", pred.outcome, outcome.var, "num.crimes")])
  #print(ordered.by.baseline[1:100, c("cell_id", "forecast_start", pred.outcome, outcome.var, "num.crimes")])
  #cat("ordered by preds:\n")
  #print(ordered.by.preds[1:100, c("cell_id", "forecast_start", pred.outcome, outcome.var, "num.crimes")])
  
  pei.by.ncells <- NULL
  for(num.cells in seq(25, 60)) {
    forecast.num <- sum(ordered.by.preds[1:num.cells, "actual"])
    global.num <- sum(ordered.by.global.preds[1:num.cells, "actual"])
    baseline.num <- sum(ordered.by.baseline[1:num.cells, "actual"])
    actual.num <- sum(ordered.by.actuals[1:num.cells, "actual"])
    
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
plot <- FALSE
for(outcome.var in names(outcome.vars)) {
  all.pei.by.cells <- adply(examine.dates, 1, function(d) {
    eval.pei(subset(cell.outcomes, forecast_start == d), outcome.var)
  }, .expand=FALSE)
  
  all.pei.by.cells <- subset(all.pei.by.cells, select=-X1)
  pei.melted <- melt(all.pei.by.cells, id.vars=c("date", "cells"))
  if(plot) {
    g <- ggplot(pei.melted, aes(x=cells, y=value, fill=variable, group=interaction(cells, variable))) + 
      geom_boxplot(outlier.size=0) + labs(title=outcome.var) + geom_jitter(aes(color=variable), alpha=0.2)
      #geom_text(data=subset(pei.melted, value < 0.5), aes(label=date), size=3) + 
    print(g)
  }
  
  med.peis <- ddply(pei.melted, .(variable, cells), summarize, med.pei=median(value))
  best.outcome.cells <- ddply(med.peis, .(variable), function(df) {
    best.cells <- df$cells[which.max(df$med.pei)]
    return(data.frame(outcome=outcome.var, best.cells=best.cells, best.pei=max(df$med.pei)))
  })
  
  perf.summary <- rbind(perf.summary, best.outcome.cells)
}
perf.summary$method <- revalue(perf.summary$variable, c("pei"="indiv", "pei.baseline"="baseline", "pei.global"="global"))
print(perf.summary)

best.choices <- ddply(perf.summary, .(outcome), function(df) {
  best.row <- df[which.max(df$best.pei), ]
  return(best.row)
})
print(best.choices[, c("outcome", "method")])

model.choices <- c(
  "outcome_num_crimes_7days_BURGLARY" = "global",
  "outcome_num_crimes_7days_ALL" = "indiv",
  "outcome_num_crimes_7days_STREET.CRIMES" = "indiv",
  "outcome_num_crimes_7days_MOTOR.VEHICLE.THEFT" = "global",
  "outcome_num_crimes_14days_BURGLARY" = "global",
  "outcome_num_crimes_14days_ALL" = "indiv",
  "outcome_num_crimes_14days_STREET.CRIMES" = "indiv",
  "outcome_num_crimes_14days_MOTOR.VEHICLE.THEFT" = "global",
  "outcome_num_crimes_30days_BURGLARY" = "global",
  "outcome_num_crimes_30days_ALL" = "baseline",
  "outcome_num_crimes_30days_STREET.CRIMES" = "baseline",
  "outcome_num_crimes_30days_MOTOR.VEHICLE.THEFT" = "indiv",
  "outcome_num_crimes_61days_BURGLARY" = "global",
  "outcome_num_crimes_61days_ALL" = "baseline",
  "outcome_num_crimes_61days_STREET.CRIMES" = "baseline",
  "outcome_num_crimes_61days_MOTOR.VEHICLE.THEFT" = "global",
  "outcome_num_crimes_91days_BURGLARY" = "global",
  "outcome_num_crimes_91days_ALL" = "baseline",
  "outcome_num_crimes_91days_STREET.CRIMES" = "baseline",
  "outcome_num_crimes_91days_MOTOR.VEHICLE.THEFT" = "global"
)

output.date <- "2017-03-01"

output.dirs <- c("Burg"="BURGLARY", "ACFS"="ALL", "SC"="STREET.CRIMES", "TOA"="MOTOR.VEHICLE.THEFT")
output.windows <- c("1MO"=7, "1WK"=14, "2MO"=30, "2WK"=61, "3MO"=91)
base.dir <- "~/repos/nij-crime-challenge/models/poisson/GARANTANALYTICS"

target.forecast <- subset(cell.outcomes, forecast_start == output.date)
projection <- CRS("+proj=lcc +lat_1=44.33333333333334 +lat_2=46 +lat_0=43.66666666666666 +lon_0=-120.5 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=ft +no_defs")
cells <- readShapePoly("~/repos/nij-crime-challenge/models/cells/cells-dim-550.shp", proj4string=projection)

for(type in names(output.dirs)) {
  for(window in names(output.windows)) {
    current.dir <- file.path(base.dir, type, window)
    dir.create(current.dir, recursive=TRUE, showWarnings=FALSE)
    
    outcome.var <- paste0("outcome_num_crimes_", output.windows[[window]] ,"days_", output.dirs[[type]])
    model.type <- model.choices[[outcome.var]]
    ncells <- subset(perf.summary, method == model.type & outcome == outcome.var)$best.cells
    ranked.forecast <- get.ranked.cells(target.forecast, model.type, outcome.var)[1:ncells, ]
    cat("Forecasting", ncells, "hotspots for", outcome.var, "\n")
    area.sqft <- sum(subset(cell.meta, id %in% ranked.forecast$cell_id)$area)
    area.sqmi <- area.sqft / (5280 * 5280)
    cat("\tTotal area: ", area.sqft, "sq ft\n")
    cat("\tTotal area: ", area.sqmi, "sq mi\n")
    cat("\tNum forecast crimes:", sum(ranked.forecast$pred), "\n")
    cat("\tShould be NA:", max(ranked.forecast$actual), "\n")
    cells@data <- data.frame(id=cells@data$id, area=round(cells@data$area, 4), 
                             hotspot=as.integer(cells@data$id %in% ranked.forecast$cell_id))
    rownames(cells@data) <- cells@data$id
    
    if(area.sqmi < 0.26 || area.sqmi > 0.71) {
      stop("bad area")
    }
    
    writeSpatialShape(cells, file.path(current.dir, paste0("GARANTANALYTICS_", type, "_", window)))
    
    handle <- file(file.path(current.dir, paste0("GARANTANALYTICS_", type, "_", window, ".prj")))
    writeLines(paste0("PROJCS[\"NAD_1983_HARN_StatePlane_Oregon_North_FIPS_3601_Feet_Intl\",",
                      "GEOGCS[\"GCS_North_American_1983_HARN\"",
                      ",DATUM[\"D_North_American_1983_HARN\"", 
                      ",SPHEROID[\"GRS_1980\",6378137.0,298.257222101]],",
                      "PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],",
                      "PROJECTION[\"Lambert_Conformal_Conic\"],",
                      "PARAMETER[\"False_Easting\",8202099.737532808],",
                      "PARAMETER[\"False_Northing\",0.0],",
                      "PARAMETER[\"Central_Meridian\",-120.5],",
                      "PARAMETER[\"Standard_Parallel_1\",44.33333333333334],",
                      "PARAMETER[\"Standard_Parallel_2\",46.0],",
                      "PARAMETER[\"Latitude_Of_Origin\",43.66666666666666],UNIT[\"Foot\",0.3048]]"), handle)
    close(handle)
  }
}
