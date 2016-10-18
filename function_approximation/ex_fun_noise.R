
z <- function(x,y) 2*x^2 - 3*y^2 + 1


make_train_set <- function(min, max, by=0.1) {
  domain <- seq(min,max, by=by)
  xy <- expand.grid(domain, domain)
  xy$z <- apply(xy,1, function(r) z(r[1], r[2]))
  xy
}


# Add uniform noise
#
# @examples
# df <- make_train_set(-10,10)
# df1 <- perturb(df)
# write.table(df1,'train_w_unif.csv', sep=',', row.names=FALSE, col.names=FALSE)
perturb <- function(df, mult=4, fr=function(a) a + runif(length(a)) - .5) {
  fn <- function(i) data.frame(x=fr(df[,1]), y=fr(df[,2]), z=df[,3])
  o <- lapply(1:mult, fn)
  do.call(rbind, o)
}


# Scaled Gaussian noise
norm_pertruber <- function(a, scale=.2) a * scale * rnorm(length(a)) - 0.5

