library(ggplot2)
library(maptools)
library(ggmap)
library(rgdal)
library(ptinpoly)
library(plyr)
library(rgeos)

projection <- CRS("+proj=lcc +lat_1=44.33333333333334 +lat_2=46 +lat_0=43.66666666666666 +lon_0=-120.5 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=ft +no_defs")

data.files <- Sys.glob("../../data/NIJ*.shp")

crimes <- NULL
for(f in data.files) {
  cat("processing", f, "\n")
  new.dat <- readShapeSpatial(f, proj4string=projection)
  if(is.null(crimes)) {
    crimes <- new.dat
  } else {
    crimes <- rbind(crimes, new.dat)
  }
}

cells <- readShapePoly("../../models/cells/cells-dim-550.shp", proj4string=projection)
crimes.by.cell <- over(crimes, cells)
crimes.by.cell$category <- crimes@data$CATEGORY
crime.counts.by.category.and.cell <- subset(ddply(crimes.by.cell, .(id, category), summarize, num.crimes=length(id)), !is.na(id))
crime.counts.by.cell <- ddply(crime.counts.by.category.and.cell, .(id), summarize, num.crimes=sum(num.crimes))

area.cells <- fortify(cells)
area.cells.meta <- merge(area.cells, crime.counts.by.cell, by="id", all.x=TRUE)
area.cells.meta[is.na(area.cells.meta$num.crimes), "num.crimes"] <- 0
area.cells.meta <- area.cells.meta[order(area.cells.meta$group, area.cells.meta$order), ]

ggplot(area.cells.meta, aes(x=long, y=lat, group=group, fill=num.crimes)) + geom_polygon()

write.model <- function(path, crime.counts, num.hotspots=30) {
  hotspot.ids <- crime.counts[order(-crime.counts$num.crimes), ][1:num.hotspots, "id"]
  cells@data$hotspot <- cells@data$id %in% hotspot.ids
  writeSpatialShape(cells, path)
}

if(dir.exists("../../models/baseline")) {
  for(timeframe in c("1MO", "1WK", "2MO", "2WK", "3MO")) {
    dir.create(paste0("../../models/baseline/ACFS/", timeframe), showWarnings=FALSE)  
    dir.create(paste0("../../models/baseline/Burg/", timeframe), showWarnings=FALSE)  
    dir.create(paste0("../../models/baseline/SC/", timeframe), showWarnings=FALSE)  
    dir.create(paste0("../../models/baseline/TOA/", timeframe), showWarnings=FALSE)  
    
    write.model(paste0("../../models/baseline/ACFS/", timeframe , "/BASEINE_ACFS_", timeframe), crime.counts.by.cell)
    write.model(paste0("../../models/baseline/ACFS/", timeframe , "/BASEINE_Burg_", timeframe), 
                subset(crime.counts.by.category.and.cell, category == "BURGLARY"))
    write.model(paste0("../../models/baseline/ACFS/", timeframe , "/BASEINE_SC_", timeframe), 
                subset(crime.counts.by.category.and.cell, category == "STREET CRIMES"))
    write.model(paste0("../../models/baseline/ACFS/", timeframe , "/BASEINE_TOA_", timeframe), 
                subset(crime.counts.by.category.and.cell, category == "MOTOR VEHICLE THEFT"))
  }
} else {
  stop("baseline model directory doesn't exist, maybe we're running from the wrong directory.")
}
