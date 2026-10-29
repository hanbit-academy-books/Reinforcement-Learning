import tensorflow as tf

# example for computing the gradient (dy/dx1,dy/dx2)
x1,x2=tf.Variable(4.0),tf.Variable(-2.0)
with tf.GradientTape() as tape:
    y=1.5*x1*x1-2.0*x2*x2+3.5*x1*x2-2.0*x2-15.0
(dy_dx1,dy_dx2)=tape.gradient(y,[x1,x2])
print('\nGradient values',f'(dy/dx1,dy/dx2)=({dy_dx1.numpy():.3f},{dy_dx2.numpy():.3f})\n')

# example for learning the parameters(weights) a, b, c from data (X,Y)
a,b,c=tf.Variable(0.0),tf.Variable(0.0),tf.Variable(0.0)
weights=[a,b,c]
X,Y=tf.constant([(0.0,0.0),(1.0,1.0),(1.0,0.0)]),tf.constant([-4.0,1.0,-2.0])
optimizer=tf.optimizers.Adam(learning_rate=0.01)

n_epoch=2000
for e in range(n_epoch):
    with tf.GradientTape() as tape:
        y_=a*X[:,0]+b*X[:,1]+c
        loss=tf.reduce_mean(tf.square(y_-Y))
    grad=tape.gradient(loss,weights)
    optimizer.apply_gradients(zip(grad,weights))

    if e%100==0:
        print(f'E {e}, Loss={loss.numpy():.4f}, weights=({a.numpy():.3f},{b.numpy():.3f},{c.numpy():.3f}')

print('converged weights',f'(a,b,c)=({a.numpy():.3f},{b.numpy():.3f},{c.numpy():.3f})')
