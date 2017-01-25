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
cells <- expand.grid(x0=factor(cell.longs), y0=factor(cell.lats))
cells$nx <- as.numeric(cells$x0)
cells$ny <- as.numeric(cells$y0)
cells$x0 <- as.numeric(as.character(cells$x0))
cells$y0 <- as.numeric(as.character(cells$y0))
cells$x1 <- cells$x0 + cell.dimension.ft
cells$y1 <- cells$y0 + cell.dimension.ft
cells$cellid <- 0:(nrow(cells)-1)
rownames(cells) <- cells$cellid

cat("Computing polygon list\n")
polygon.list <- alply(cells, 1, function(row) {
  spec <- rbind(c(row$x0, row$y0), 
        c(row$x1, row$y0),
        c(row$x1, row$y1),
        c(row$x0, row$y1),
        c(row$x0, row$y0))
  Polygons(list(Polygon(spec, hole=FALSE)), row$cellid)
})
polygon.struct <- SpatialPolygonsDataFrame(SpatialPolygons(polygon.list, proj4string=proj), cells)
cat("Intersecting polygons with study area\n")
valid.cells <- gIntersection(polygon.struct, study.area, byid=TRUE)

cat("Area of districts: ", gArea(districts) / (5280 * 5280), "\n")
cat("Area of valid cells: ", gArea(valid.cells)  / (5280 * 5280), "\n")

cat("Converting polygons to data frame\n")
area.cells <- fortify(polygon.struct)

ggplot(area.districts, aes(x=long, y=lat, group=group)) + geom_polygon() + 
    geom_polygon(data=valid.cells, aes(x=long, y=lat, group=group), alpha=0.5, fill="red", color="black", size=0.5)

cat("Renaming cells\n")
poly.id.map <- list()
old.ids <- laply(strsplit(laply(valid.cells@polygons, function(p) p@ID), "\\s"), function(v) v[1])
new.ids <- as.character(0:(length(valid.cells@polygons)-1))
poly.id.map[old.ids] <- new.ids
valid.cells.data <- subset(cells, cellid %in% old.ids)
valid.cells.data$id <- unlist(poly.id.map[as.character(valid.cells.data$cellid)], use.names=FALSE)
valid.cells.data$cellid <- NULL

# compute adjacencies based on nx and ny
valid.cells.data$coord <- paste0(valid.cells.data$nx, ",", valid.cells.data$ny)
coord.index <- split(valid.cells.data$id, valid.cells.data$coord)
get.adj.id <- function(offsetx, offsety) {
  laply(coord.index[paste0(valid.cells.data$nx+offsetx, ",", valid.cells.data$ny+offsety)], 
        function(v) if(is.null(v)) NA else v)
}
valid.cells.data$idwest <- get.adj.id(-1, 0)
valid.cells.data$idnorth <- get.adj.id(0, 1)
valid.cells.data$ideast <- get.adj.id(1, 0)
valid.cells.data$idsouth <- get.adj.id(0, -1)
valid.cells.data$idnorthwest <- get.adj.id(-1, 1)
valid.cells.data$idnortheast <- get.adj.id(1, 1)
valid.cells.data$idsouthwest <- get.adj.id(-1, -1)
valid.cells.data$idsoutheast <- get.adj.id(1, -1)
rownames(valid.cells.data) <- valid.cells.data$id

valid.cells.renamed <- spChFIDs(valid.cells, valid.cells.data$id)
cell.areas <- laply(valid.cells.renamed@polygons, function(p) {
  sum(laply(p@Polygons, function(p) { if (p@hole) -p@area else p@area }))
})
valid.cells.data$area <- cell.areas

valid.cells.df <- SpatialPolygonsDataFrame(valid.cells.renamed, valid.cells.data)
writePolyShape(valid.cells.df, paste0("../../models/cells/cells-dim-", cell.dimension.ft))

# cluster the cells by crime counts
cat("Building cell clusters\n")
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

crimes.by.cell <- over(crimes, valid.cells.df)
crimes.by.cell$category <- crimes@data$CATEGORY
crime.counts.by.cell <- ddply(crimes.by.cell, .(id), summarize, num.crimes=length(id))

library(GGally) 
valid.cells.center <- ldply(valid.cells.df@polygons, function(p) {
  a <- p@labpt
  data.frame(x=a[1], y=a[2], id=p@ID)
})
valid.cells.center <- merge(valid.cells.center, valid.cells.df@data, by="id")
crime.meta <- merge(valid.cells.center, crime.counts.by.cell, by="id", all.x=TRUE)
crime.meta[is.na(crime.meta$num.crimes), "num.crimes"] <- 0
ggplot(crime.meta, aes(x=x, y=y, color=num.crimes)) + 
  geom_point(stroke=0, size=3) + scale_color_continuous(low="blue", high="red")
rownames(crime.meta) <- crime.meta$id
write.table(subset(crime.meta, select=-coord), 
            paste0("../../models/cells/cells-dim-", cell.dimension.ft, "-meta.csv"), 
            row.names=FALSE, quote=FALSE, sep=",")

# validate the IDs by plotting a few cells and their adjacent nodes
id.sample <- sample(crime.meta$id, 10)
for(cur.id in id.sample) {
  cur.row <- crime.meta[cur.id, ]
  crime.meta$type <- NA
  crime.meta[cur.id, "type"] <- "ref"
  if(!is.na(cur.row$ideast))
    crime.meta[cur.row$ideast, "type"] <- "E"
  if(!is.na(cur.row$idwest))
    crime.meta[cur.row$idwest, "type"] <- "W"
  if(!is.na(cur.row$idnorth))
    crime.meta[cur.row$idnorth, "type"] <- "N"
  if(!is.na(cur.row$idsouth))
    crime.meta[cur.row$idsouth, "type"] <- "S"
  if(!is.na(cur.row$idnorthwest))
    crime.meta[cur.row$idnorthwest, "type"] <- "NW"
  if(!is.na(cur.row$idnortheast))
    crime.meta[cur.row$idnortheast, "type"] <- "NE"
  if(!is.na(cur.row$idsouthwest))
    crime.meta[cur.row$idsouthwest, "type"] <- "SW"
  if(!is.na(cur.row$idsoutheast))
    crime.meta[cur.row$idsoutheast, "type"] <- "SE"
  
  g <- ggplot(crime.meta, aes(x=x, y=y, color=type)) + geom_point() + labs(title=cur.id)
  print(g)
}
