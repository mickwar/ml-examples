f = function(x, q)
    ifelse(x > 0, q*x, (q-1)*x)

png("loss.png", width = 640)
x = seq(-3, 3, length = 101)
plot(0, type='n', xlim = range(x), ylim = c(0, 1.5),
    xlab = expression(y - hat(y)),
    ylab = "Loss", axes = FALSE, cex.lab = 1.5)
points(x, f(x, 0.5), type = 'l', lwd = 3, col = 'black')
points(x, f(x, 0.1), type = 'l', lwd = 3, col = 'red')
points(x, f(x, 0.7), type = 'l', lwd = 3, col = 'blue')
text(-2.5, 0.87, "q=0.7", cex = 1.5)
text(-2.18, 1.27, "q=0.5", cex = 1.5)
text(-1.22, 1.40, "q=0.1", cex = 1.5)
axis(1)
axis(2)
dev.off()

