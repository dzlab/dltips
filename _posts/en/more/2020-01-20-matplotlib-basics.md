---
layout: post

title: Matplotlib Basics

tip-number: 15
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn you some basic plotting with Matplotlib to quickly visualize your data.
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

Matplotlib is a very popular plotting library for Python. It is very powerful as it can be used to generate customized and high quality plotting. It is very important to master the basics of plotting with Matplotlib as this will allow you to better understand your data (tabular, image or text) via visualization.

Basic Matplotlib example
```python
import matplotlib.pyplot as plt
# Prepare data
x = [1, 2, 3, 4]
y = [15, 23 , 39, 46]
# Create plot
fig = plt.figure()
# Plot
ax = fig.add_subplot(111)
# Customize plot
ax.plot(x, y, color='lightblue', linewidth=3)
ax.scatter([2, 4, 6], [9, 18, 27], color='darkgreen', marker='^')
ax.set_xlim(1, 7)
# Save plot
plt.savefig('plot.png')
# Show plot
plt.show()
```

## 1. Prepare data
Numpy can be used to quickly generate data.
```python
import numpy as np
# 1D arrays
x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)
# 2D arrays
x = 2 * np.random.random((10, 10))
y = 3 * np.random.random((10, 10))
U, V = np.mgrid[-3:3:100j, -3:3:100j]
W = -1 - U**2 + V

# Images
from matplotlib.cbook import get_sample_data
img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))
```

## 2. Create plot
In Matplotlib, all plotting is done with respect to an Axes.
A `subplot`, which is an axes on a grid system, will do the job in most cases.

```python
import matplotlib.pyplot as plt

# Figure
fig1 = plt.figure()
fig2 = plt.figure(figsize=plt.figaspect(2.0))

# Axes
fig1.add_axes()
ax1 = fig1.add_subplot(221) # row-col-num
ax2 = fig1.add_subplot(212)
fig2, axes1 = plt.subplots(nrows=2,ncols=2)
fig3, axes2 = plt.subplots(ncols=3)
```

## 3. Plot
Matplot has a lot plotting helper functions for 1D or 2D data

### 3.1. 1D Data plotting
```python
# Draw points with lines or markers connecting them
lines = ax.plot(x,y)
# Draw unconnected points, scaled or colored
ax.scatter(x,y)
# Plot vertical rectangles (constant width)
axes[0,0].bar([1,2,3],[3,4,5])
# Plot horiontal rectangles (constant height)
axes[1,0].barh([0.5,1,2.5],[0,1,2])
# Draw a horizontal line across axes
axes[1,1].axhline(0.45)
# Draw a vertical line across axes
axes[0,1].axvline(0.65)
# Draw filled polygons
ax.fill(x,y,color='blue')
# Fill between y-values and 0
ax.fill_between(x,y,color='yellow')
```

### 3.2. 2D Data plotting
Draw data color mapped or RGB
```python
fig, ax = plt.subplots()
im = ax.imshow(img, cmap='gist_earth', interpolation='nearest', vmin=-2, vmax=2)
```

### 3.3. Vector Fields
```python
# Add an arrow to the axes
axes[0,1].arrow(0,0,0.5,0.5)
# Plot a 2D field of arrows
axes[1,1].quiver(y,z)
# Plot 2D vector fields
axes[0,1].streamplot(X,Y,U,V)
```

### 3.4. Data Distributions
Quickly visualize the distribution graphs of 1D data
```python
# Plot a histogram
ax1.hist(y)
# Make a box and whisker plot
ax2.boxplot(y)
# Make a violin plot
ax3.violinplot(z)
```

Distribution plotting for 2D data
```python
# Pseudocolor plot of 2D array
axes[0].pcolor(data)
# Pseudocolor plot of 2D array
axes[0].pcolormesh(data)
# Plot contours
CS = plt.contour(Y,X,U)
# Plot filled contours
axes[2].contourf(data1)
# Label a contour plot
axes[2]= ax.clabel(CS)
```

## 4. Customize Plot
Matplotlib let you customize a plot bys setting the colors, markers, labels, etc.

### 4.1. Colors, Color Bars & Color Maps
```python
plt.plot(x, x, x, x**2, x, x**3)
ax.plot(x, y, alpha = 0.4)
ax.plot(x, y, c='k')
fig.colorbar(im, orientation='horizontal')
im = ax.imshow(img,cmap='seismic')
```

### 4.2. Markers
```python
fig, ax = plt.subplots()
ax.scatter(x,y,marker=".")
ax.plot(x,y,marker="o")
```

### 4.3. Linestyles
```python
plt.plot(x,y,linewidth=4.0)
plt.plot(x,y,ls='solid')
plt.plot(x,y,ls='--')
plt.plot(x,y,'--',x**2,y**2,'-.')
plt.setp(lines,color='r',linewidth=4.0)
```

### 4.4. Text & Annotations
```python
ax.text(1, -2.1, 'Title', style='italic')
ax.annotate(
  "Sine", xy=(8, 0), xycoords='data', xytext=(10.5, 0), textcoords='data',
  arrowprops=dict(arrowstyle="->", connectionstyle="arc3")
)
```

### 4.5. Mathtext
```python
plt.title(r'$sigma_i=15$', fontsize=20)
```

### 4.6. Limits, Legends & Layouts

#### 4.6.1. Limits & Autoscaling
```python
# Add padding to a plot
ax.margins(x=0.0,y=0.1)
# Set the aspect ratio of the plot to 1
ax.axis('equal')
# Set limits for x-and y-axis
ax.set(xlim=[0,10.5], ylim=[-1.5,1.5])
# Set limits for x-axis
ax.set_xlim(0,10.5)
```

#### 4.6.2. Legends
```python
# Set a title and x-and y-axis labels
ax.set(title='An Example Axes', ylabel='Y-Axis',xlabel='X-Axis')
# No overlapping plot elements
ax.legend(loc='best')
```

#### 4.6.3. Ticks
```python
ax.xaxis.set(ticks=range(1,5), ticklabels=[3,100,-12,"foo"], direction='inout', length=10))
```

#### 4.6.4. Subplot Spacing
```python
fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1))
fig.tight_layout()
```


#### 4.6.5. Axis Spines
```python
# Make the top axis line for a plot invisible
ax.spines['top'=].set_visible(False)
# Move the bottom axis line outward
ax.spines['bottom'].set_position(('outward',10))
```


## 5. Save Plot
```python
# Save figure
plt.savefig('image.png')
# Save figure with transparency
plt.savefig('image.png', transparent=True)
```

## 6. Show Plot
```python
plt.show()
```

##  7. Close & Clear
```python
plt.cla()
plt.clf()
plt.close()
```