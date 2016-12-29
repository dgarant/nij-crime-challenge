library(ggplot2)
library(maptools)
library(ggmap)
library(rgdal)
library(ptinpoly)
library(raster)
library(plyr)
library(rgeos)

# 550 -> 302500sqft cells, about 14K cells
cell.dimension.ft <- 550

proj <- CRS("+proj=lcc +lat_1=44.33333333333334 +lat_2=46 +lat_0=43.66666666666666 +lon_0=-120.5 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=ft +no_defs")
districts <- readShapePoly("../../data/Portland_Police_Districts.shp", proj4string=proj)
study.area <- unionSpatialPolygons(districts, rep(1, length(districts@polygons)))
study.area.df <- fortify(study.area)
ggplot(study.area.df, aes(x=long, y=lat, group=group)) + geom_polygon()

# area in square feet
#sqfts <- sapply(slot(districts, "polygons"), slot, "area")
#print(sum(sqfts) / (5280 * 5280))
area.districts <- fortify(districts)

# grid out the city
long.upper <- max(area.districts$long)
long.lower <- min(area.districts$long)
lat.upper <- max(area.districts$lat)
lat.lower <- min(area.districts$lat)

cell.lats <- seq(from=lat.lower, to=lat.upper+cell.dimension.ft, by=cell.dimension.ft)
cell.longs <- seq(from=long.lower, to=long.upper+cell.dimension.ft, by=cell.dimension.ft)
cells <- expand.grid(x0=cell.longs, y0=cell.lats)
cells$x1 <- cells$x0 + cell.dimension.ft
cells$y1 <- cells$y0 + cell.dimension.ft
cells$cellid <- 1:nrow(cells)
polygon.list <- alply(cells, 1, function(row) {
  spec <- rbind(c(row$x0, row$y0), 
        c(row$x1, row$y0),
        c(row$x1, row$y1),
        c(row$x0, row$y1),
        c(row$x0, row$y0))
  Polygons(list(Polygon(spec, hole=FALSE)), row$cellid)
})
polygon.struct <- SpatialPolygons(polygon.list, cells$cellid, proj4string=proj)
valid.cells <- gIntersection(polygon.struct, study.area, byid=TRUE)
area.cells <- fortify(polygon.struct)

gArea(districts) / (5280 * 5280)
gArea(valid.cells)  / (5280 * 5280)

ggplot(area.districts, aes(x=long, y=lat, group=group)) + geom_polygon() + 
    geom_polygon(data=valid.cells, aes(x=long, y=lat, group=group), alpha=0.5, fill="red", color="black", size=0.5)

poly.ids <- as.character(0:(length(valid.cells)-1))
valid.cells.renamed <- spChFIDs(valid.cells, poly.ids)
cell.areas <- laply(valid.cells.renamed@polygons, function(p) {
  sum(laply(p@Polygons, function(p) { if (p@hole) -p@area else p@area }))
})

valid.cells.df <- SpatialPolygonsDataFrame(valid.cells.renamed, 
                                           data.frame(id=poly.ids, area=cell.areas, row.names=poly.ids))
writePolyShape(valid.cells.df, paste0("../../models/cells/cells-dim-", cell.dimension.ft))


adjacencies <- gTouches(valid.cells.df, byid=TRUE, returnDense=FALSE)
adj.df <- adply(names(adjacencies), 1, function(name) { 
  cur.adjs <- adjacencies[[name]]
  if(length(cur.adjs) == 0) {
    return(NULL)
  } else {
    return(data.frame(a=as.integer(name), b=cur.adjs))
  }
})
adj.df <- subset(adj.df, select=-X1)
adj.df <- adj.df[!duplicated(adj.df), ]
write.csv(adj.df, paste0("../../models/cells/cells-dim-", cell.dimension.ft, "-adjacencies.csv"))
write.csv(data.frame(id=poly.ids), paste0("../../models/cells/cells-dim-", cell.dimension.ft, "-ids.csv"))
