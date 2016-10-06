
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
# df1 <- perturb_unif(df)
# write.table(df1,'train_w_unif.csv', sep=',', row.names=FALSE, col.names=FALSE)
perturb_unif <- function(df, mult=4) {
  fr <- function(a,n) a + runif(n) - .5
  fz <- function(z) { names(z) <- NULL; z }
  fn <- function(r) data.frame(x=fr(r[1], mult), y=fr(r[2], mult), z=fz(r[3]))
  o <- apply(df, 1, fn)
  do.call(rbind, o)
}


# Add gaussian noise
perturb_norm <- function(xy) {

}
