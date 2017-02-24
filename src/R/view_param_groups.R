library(digest)
library(ggplot2)

cell.meta <- read.csv("~/repos/nij-crime-challenge/models/cells/cells-dim-550-meta.csv")

groups <- read.csv("~/repos/nij-crime-challenge/models/mrf/param_groups.csv")
c2 <- merge(cell.meta, groups, by.x="id", by.y="cellid", suffixes=c("", ".g"), all.x=TRUE)
# using a diverse set of colors for adjaceny groups
group.hash <- factor(floor((tan(c2$group) * 100000)) %% 20)

ungrouped <- subset(c2, is.na(group))
cat("ungrouped (", nrow(ungrouped) , "total):\n")
print(summary(ungrouped$num.crimes))
print(head(ungrouped[order(-ungrouped$num.crimes), ]))

#ggplot(c2, aes(x=x, y=y, color=num.crimes)) + geom_point() + guides(fill="none")
g <- ggplot(c2, aes(x=x, y=y, group=group, color=group.hash)) + geom_point() + guides(fill="none")
print(g)
g <- ggplot(ungrouped, aes(x=x, y=y, color=num.crimes)) + geom_point() + guides(fill="none")
print(g)
