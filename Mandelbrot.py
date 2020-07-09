#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[10]:


def mandelbrot (h,w,maxit=20):
    #returns an image of the Mandelbrot fractal of size (h,w)
    y,x=np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)
    
    
    for i in range(maxit):
        z= z**2 + c
        diverge = z*np.conj(z) > 2**2    #who is diverging
        div_now = diverge & (divtime==maxit)  #who is diverging now
        divtime[div_now] = i                 #note when
        z[diverge] = 2                    #avoid diverging too much
        
    return divtime
        
    


# In[12]:


plt.imshow(mandelbrot(400,400))


# In[28]:


from PIL import Image 
from numpy import complex, array 
import colorsys 


# In[14]:


# setting the width of the output image as 1024 
WIDTH = 1024


# In[15]:


# a function to return a tuple of colors 
# as integer value of rgb 
def rgb_conv(i): 
    color = 255 * array(colorsys.hsv_to_rgb(i / 255.0, 1.0, 0.5)) 
    return tuple(color.astype(int)) 


# In[16]:


# function defining a mandelbrot 
def mandelbrot(x, y): 
    c0 = complex(x, y) 
    c = 0
    for i in range(1, 1000): 
        if abs(c) > 2: 
            return rgb_conv(i) 
        c = c * c + c0 
    return (0, 0, 0) 


# In[17]:


# creating the new image in RGB mode 
img = Image.new('RGB', (WIDTH, int(WIDTH / 2))) 
pixels = img.load() 


# In[18]:


for x in range(img.size[0]): 
  
    # displaying the progress as percentage 
    print("%.2f %%" % (x / WIDTH * 100.0))  
    for y in range(img.size[1]): 
        pixels[x, y] = mandelbrot((x - (0.75 * WIDTH)) / (WIDTH / 4), 
                                      (y - (WIDTH / 4)) / (WIDTH / 4)) 


# In[39]:


# to display the created fractal after  
# completing the given number of iterations 
plt.figure()
plt.imshow(img)


# In[40]:


# Mandelbrot fractal 
# FB - 201003254 
from PIL import Image 
  
# drawing area 
xa = -2.0
xb = 1.0
ya = -1.5
yb = 1.5
  
# max iterations allowed 
maxIt = 255 
  
# image size 
imgx = 512
imgy = 512
image = Image.new("RGB", (imgx, imgy)) 
  
for y in range(imgy): 
    zy = y * (yb - ya) / (imgy - 1)  + ya 
    for x in range(imgx): 
        zx = x * (xb - xa) / (imgx - 1)  + xa 
        z = zx + zy * 1j
        c = z 
        for i in range(maxIt): 
            if abs(z) > 2.0: break
            z = z * z + c 
        image.putpixel((x, y), (i % 4 * 64, i % 8 * 32, i % 16 * 16)) 
  
plt.figure()
plt.imshow(image)


# In[ ]:




