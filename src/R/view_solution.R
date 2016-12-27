library(ggplot2)
library(maptools)
library(ggmap)
library(rgdal)
library(ptinpoly)
library(raster)
library(plyr)
library(rgeos)
library(stringr)

proj <- CRS("+proj=lcc +lat_1=44.33333333333334 +lat_2=46 +lat_0=43.66666666666666 +lon_0=-120.5 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=ft +no_defs")

solution <- readShapePoly("../../models/baseline/ACFS/1WK/BASEINE_Burg_1WK.shp", proj4string=proj)
solution@data$Id <- rownames(solution@data)
num.forecast.hot <- sum(solution@data$hotspot == 1)

districts <- readShapePoly("../../data/Portland_Police_Districts.shp", proj4string=proj)


ref.period <- readShapeSpatial("../../data/NIJ2016_OCT01_OCT31.shp", proj4string = proj)
ref.dates <- paste0("2016-10-", str_pad(1:7, 2, pad="0"))
ref.period.filtered <- ref.period[ref.period$occ_date %in% as.Date(ref.dates), ]
# filter to study area
ref.period.filtered <- raster::intersect(ref.period.filtered, districts)

#hotspots <- solution[solution$hotspot == 1, ]
join.actuals <- over(ref.period.filtered, solution)

actual.cell.counts <- ddply(join.actuals, .(Id), summarize, num.crimes=length(Id), area=area[1])
actual.hot.ids <- actual.cell.counts[order(-actual.cell.counts$num.crimes), ][1:num.forecast.hot, "Id"]

solution.meta <- solution@data
solution.meta$Id <- rownames(solution@data)
solution.meta$actual.hot <- solution.meta$Id %in% actual.hot.ids
solution.meta <- merge(solution.meta, subset(actual.cell.counts, select=-area), by="Id", all.x=TRUE)
solution.meta[is.na(solution.meta$num.crimes), "num.crimes"] <- 0

cfactor <- (5280 * 5280)

cell.areas <- ldply(solution@polygons, function(p) {
  areas <- laply(p@Polygons, function(p) { if (p@hole) -p@area else p@area })
  data.frame(id=p@ID, area=sum(areas), trunc.area = sum(floor(areas)))
})

study.area.sqft <- gArea(solution)
cat("Smallest cell area: ", min(cell.areas$area), " (dimension ", sqrt(min(cell.areas$area)) , ")\n")
cat("Median cell area: ", median(cell.areas$area), " (dimension ", sqrt(median(cell.areas$area)) , ")\n")
cat("Total area (method 1): ", sum(cell.areas$area) / cfactor, "\n")
cat("Total area (method 2): ", sum(cell.areas$trunc.area / cfactor), "\n")
cat("Total area (method 3): ", sum(solution@data$area / cfactor), "\n")
cat("Total area (method 4): ", gArea(solution) / cfactor, "\n")
cat("Total hotspot area: ", sum(subset(solution@data, hotspot==1)$area / cfactor), "\n")
cat("Num hotspots: ", num.forecast.hot, "\n")

forecasted.hotspots <- subset(solution.meta, hotspot == 1)
actual.hotspots <- subset(solution.meta, actual.hot)
pai <- (sum(forecasted.hotspots$num.crimes) / nrow(ref.period.filtered)) / (sum(forecasted.hotspots$area) / study.area.sqft)
pai.best <- (sum(actual.hotspots$num.crimes) / nrow(ref.period.filtered)) / (sum(actual.hotspots$area) / study.area.sqft)
pei <- pai / pai.best
cat("PAI: ", pai, "\n")
cat("PAI*: ", pai.best, "\n")
cat("PEI: ", pei, "\n")

area.solution <- fortify(solution)
area.solution$Id <- area.solution$id
area.solution.meta <- merge(area.solution, solution.meta, by="Id", all.x=TRUE)
area.solution.meta <- area.solution.meta[order(area.solution.meta$group, area.solution.meta$order), ]

ggplot(area.solution.meta, aes(x=long, y=lat)) + 
  geom_polygon(aes(group=group, fill=paste0(actual.hot, hotspot)))
  #geom_point(data=ref.period.filtered@data, 
   #          aes(x=x_coordina, y=y_coordina, group=NULL, fill=NULL))
  