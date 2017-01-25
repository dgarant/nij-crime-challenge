library(ggplot2)
library(maptools)
library(ggmap)
library(rgdal)
library(ptinpoly)
library(plyr)
library(rgeos)

projection <- CRS("+proj=lcc +lat_1=44.33333333333334 +lat_2=46 +lat_0=43.66666666666666 +lon_0=-120.5 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=ft +no_defs")

#TODO: add projection, hotspot 0/1

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
  
  handle <- file(paste0(path, ".prj"))
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

if(dir.exists("../../models/baseline")) {
  for(timeframe in c("1MO", "1WK", "2MO", "2WK", "3MO")) {
    dir.create(paste0("../../models/baseline/ACFS/", timeframe), showWarnings=FALSE)  
    dir.create(paste0("../../models/baseline/Burg/", timeframe), showWarnings=FALSE)  
    dir.create(paste0("../../models/baseline/SC/", timeframe), showWarnings=FALSE)  
    dir.create(paste0("../../models/baseline/TOA/", timeframe), showWarnings=FALSE)  
    
    write.model(paste0("../../models/baseline/ACFS/", timeframe , "/BASELINE_ACFS_", timeframe), crime.counts.by.cell)
    write.model(paste0("../../models/baseline/Burg/", timeframe , "/BASELINE_Burg_", timeframe), 
                subset(crime.counts.by.category.and.cell, category == "BURGLARY"))
    write.model(paste0("../../models/baseline/SC/", timeframe , "/BASELINE_SC_", timeframe), 
                subset(crime.counts.by.category.and.cell, category == "STREET CRIMES"))
    write.model(paste0("../../models/baseline/TOA/", timeframe , "/BASELINE_TOA_", timeframe), 
                subset(crime.counts.by.category.and.cell, category == "MOTOR VEHICLE THEFT"))
  }
} else {
  stop("baseline model directory doesn't exist, maybe we're running from the wrong directory.")
}
