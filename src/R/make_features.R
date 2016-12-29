library(ggplot2)
library(maptools)
library(ggmap)
library(rgdal)
library(ptinpoly)
library(plyr)
library(rgeos)

cell.dimension.ft <- 550

proj <- CRS("+proj=lcc +lat_1=44.33333333333334 +lat_2=46 +lat_0=43.66666666666666 +lon_0=-120.5 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=ft +no_defs")
cells <- readShapePoly(paste0("../../models/cells/cells-dim-", cell.dimension.ft, ".shp"), proj4string=proj)

data.files <- Sys.glob("../../data/NIJ*.shp")

crimes <- NULL
for(f in data.files) {
  cat("processing", f, "\n")
  new.dat <- readShapeSpatial(f, proj4string=proj)
  if(is.null(crimes)) {
    crimes <- new.dat
  } else {
    crimes <- rbind(crimes, new.dat)
  }
}

crime.cell <- over(crimes, cells)
crime.meta <- subset(cbind(crimes@data, crime.cell), !is.na(id))
crime.meta$CATEGORY2 <- paste0(crime.meta$CATEGORY, "-", crime.meta$CALL_GROUP)
write.csv(crime.meta, file=gzfile(paste0("../../features/raw_crimes_cells_", cell.dimension.ft, ".csv.gz")))
#counts.by.cell <- ddply(crime.meta, .(id, CATEGORY, CALL_GROUP, CASE_DESC, occ_date), num_crimes=length(id))
