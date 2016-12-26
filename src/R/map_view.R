library(ggplot2)
library(maptools)
library(ggmap)
library(rgdal)
library(ptinpoly)
library(plyr)
library(rgeos)

projection <- CRS("+proj=lcc +lat_1=44.33333333333334 +lat_2=46 +lat_0=43.66666666666666 +lon_0=-120.5 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=ft +no_defs")

data.files <- Sys.glob("../../data/NIJ*.shp")

shapes <- NULL
for(f in data.files) {
  cat("processing", f, "\n")
  new.dat <- readShapeSpatial(f, proj4string=projection)
  if(is.null(shapes)) {
    shapes <- new.dat
  } else {
    shapes <- rbind(shapes, new.dat)
  }
}
shapes.transform <- spTransform(shapes, CRS("+proj=longlat +ellps=WGS84"))

districts <- readShapePoly("../../data/Portland_Police_Districts.shp", proj4string=projection)
geo.districts <- spTransform(districts, CRS("+proj=longlat +ellps=WGS84"))
area.districts1 <- fortify(geo.districts)
district.centers <- gCentroid(geo.districts, byid=TRUE)
district.meta <- cbind(data.frame(district=as.character(geo.districts@data$DISTRICT)), as.data.frame(district.centers))
district.names <- data.frame(id=0:(nrow(geo.districts@data)-1), district=as.character(geo.districts@data$DISTRICT))
area.districts2 <- merge(area.districts1, district.names, by="id", all.x=TRUE)
area.districts <- area.districts2[order(area.districts2$group, area.districts2$order), ]

crime.district <- over(shapes, districts)$DISTRICT
crime.coords <- cbind(data.frame(district=crime.district, category=shapes@data$CATEGORY), as.data.frame(coordinates(shapes.transform)))

map.image <- get_map(location=c(lon=-122.675968, lat=45.516478), 
                     color="color", zoom=11, maptype="roadmap")

# points overlaid on police districts:
ggmap(map.image) + geom_polygon(data=area.districts, aes(x=long, y=lat, group=group), alpha=0.9, color="white", size=1) + 
  geom_point(data=crime.coords, aes(x=coords.x1, y=coords.x2, color=category), alpha=0.5) + 
  guides(color=guide_legend(override.aes = list(size=10, alpha=1))) +
  geom_label(data=district.meta, aes(x=x, y=y, label=district), size=5, color="red") +
  coord_cartesian(x=c(-122.85, -122.47), ylim=c(45.42, 45.67))

# relative risk of particular crimes within each district
crime.table <- subset(ddply(crime.coords, .(district, category), summarize, num.crimes=length(coords.x1)), !is.na(district))
crime.table <- merge(crime.table, ddply(crime.table, .(district), summarize, total.in.district=sum(num.crimes)))
crime.table$p <- crime.table$num.crimes / crime.table$total.in.district
crime.table.wide <- reshape(crime.table, idvar=c("district"), timevar="category", direction="wide")
colnames(crime.table.wide) <- make.names(colnames(crime.table.wide))

area.districts.probs1 <- merge(area.districts, crime.table.wide, by="district")
area.districts.probs <- area.districts.probs1[order(area.districts.probs1$group, area.districts.probs1$order), ]

for(feature in c("STREET CRIMES", "BURGLARY", "MOTOR VEHICLE THEFT")) {
  name <- paste0("p.", make.names(feature))
  g <- ggmap(map.image) + geom_polygon(data=area.districts.probs, aes_string(x="long", y="lat", group="group", fill=name), alpha=0.9, color="black", size=1) + 
    #geom_label(data=district.meta, aes(x=x, y=y, label=district), size=5, color="black") +
    scale_fill_continuous(low="white", high="red") + labs(title=feature) + guides(fill="none")
    coord_cartesian(x=c(-122.85, -122.47), ylim=c(45.42, 45.67))
  print(g)
}