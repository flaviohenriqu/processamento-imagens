#-------------------------------------------------------------#
#   Trabalho 02 - Processamento de Imagens 
#   Flávio Henrique Andrade dos Santos - 201020002040
#
#-------------------------------------------------------------#

from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import operator
import os

#limpa tela
os.system('clear')


def process(img):

    B = imreadgray(imread(img)) # passo 1
    


def segmentation(path):

    with open(path) as f:
        for line in f:
            process(line);

    





#-------------------------------------------------------------#
#    
#           Trabalho 01 - Processamento de Imagens
#
#-------------------------------------------------------------#

#função responsável por ler imagens
def imread(image):
    img = Image.open(image);

    return img;


#abrir e exibir uma imagem colorida


#função que retorna o numero de canais da imagem
def nchannels(img):
    
    return len(img.getbands());

#função que retorna um vetor com o tamanho da imagem
def size(img):
   
    (width, height) = img.size;

    vetSize = [width, height];

    return vetSize;

#função que converte imagem para tons de cinza utilizando os pesos especificados
def imreadgray(img):
    
    if img.mode == "L":
        return img

    arrImg = np.asarray(img); 
    r, g, b = arrImg[:,:,0], arrImg[:,:,1], arrImg[:,:,2];
    arrGray = ((0.299 * r) + (0.587 * g) + (0.114 * b));

    gray = Image.fromarray(np.uint8(arrGray));
    
    return gray;


#função thresh
def thresh(img, lim):
    arr = np.asarray(img.point(lambda i: i > lim and 255));
    result = Image.fromarray(np.uint8(arr));
    
    return result;

#função imshow
def imshow(img):
    if(img.mode == "L"):
        plt.imshow(img, cmap=plt.cm.gray);
        
    img.show();
    return;

#função que retorna a imagem negativa
def invert(img):

    return img.point(lambda p: 255 - p);

#função maskBlur
def maskBlur():
    mat = np.asarray([[1,2,1], [2,4,2], [1,2,1]]);
    divisor = 1.0/16;
    arr = divisor * mat;
    return arr;


#função seSquare3
def seSquare3():
    arr = np.asarray([[1,1,1], [1,1,1], [1,1,1]]);
    return arr;

#função seCross3
def seCross3():
    arr = np.asarray([[0,1,0], [1,1,1], [0,1,0]]);
    return arr;

#função hist
def hist(img):
    
    sizeWH = size(img);
    if(img.mode == "L"):
        Lmax = np.amax(img);
    
        arr = np.asarray(img);
    
        H = np.zeros(Lmax, dtype=int);
    
        for x in range(sizeWH[1] - 1):
            for y in range(sizeWH[0] - 1):
                H[arr[x,y]-1] = H[arr[x,y]-1] + 1;
    else:
        arrRGB = np.asarray(img); 
        r, g, b = arrRGB[:,:,0], arrRGB[:,:,1], arrRGB[:,:,2];

        Rmax = r.max();
        Gmax = g.max();
        Bmax = b.max();

        Hr = np.zeros(Rmax, dtype=int);
        Hg = np.zeros(Gmax, dtype=int);
        Hb = np.zeros(Bmax, dtype=int);

        for x in range(sizeWH[1] - 1):
            for y in range(sizeWH[0] - 1):
                Hr[r[x,y]-1] = Hr[r[x,y]-1] + 1;
                Hg[g[x,y]-1] = Hg[g[x,y]-1] + 1;
                Hb[b[x,y]-1] = Hb[b[x,y]-1] + 1;

        H = np.asarray([Hr,Hg,Hb]);
        
    return H;

#função showhist
def showhist(hist, Bin=1):
    tamCls = Bin;
    
    if(len(hist) == 3):
        #RGB
        tamHistR, tamHistG, tamHistB = len(hist[0]), len(hist[1]), len(hist[2]);
        plt.bar(np.arange(tamHistR),hist[0],tamCls,color="red");
        plt.bar(np.arange(tamHistG),hist[1],tamCls,color="green");
        plt.bar(np.arange(tamHistB),hist[2],tamCls,color="blue");
        
    else:
        #L
        tamHist = len(hist);
        
        plt.bar(np.arange(tamHist),hist,tamCls);

    plt.show();
    
    return;

#função contrast
def contrast(im,r=1.,m=0.):
    f = np.asarray(im);
    g = ((r*(f - m)) + m);

    return Image.fromarray(np.uint8(g));

#função histeq
def histeq(img):

    h = img.histogram();
    lut = [];

    for b in range(0, len(h), 256):
        step = reduce(operator.add, h[b:b+256]) / 255;

        n = 0;
        for i in range(256):
            lut.append(n / step);
            n = n + h[i+b];

    g = img.point(lut);

    return g;
#    return ImageOps.equalize(img);


#função convolve
def convolve(img, mask=[[1.,1.,1.], [1.,1.,1.], [1.,1.,1.]]):
    mask = np.asarray(mask);
    
    x1, y1 = mask.shape;
    x1 /= 2;
    y1 /= 2;
    f = np.asarray(img);
    sizeMN = size(img);
    g = np.zeros(f.shape);

    if(img.mode == "L"):
        for x in range(sizeMN[0] - 1):
            for y in range(sizeMN[1] - 1):
                soma = 0;
                for i in range(-x1,x1+1):
                    for j in range(-y1,y1+1):
                        fx = x - i;
                        if(fx < 0):
                            fx = 0;
                        fy = y - j;
                        if(fy >= sizeMN[0]):
                            fy = sizeMN[0] - 1;
                            
                        soma += mask[i,j]*f[fx,fy];
                g[x,y] = soma;
    else:
        fr, fg, fb = f[:,:,0],f[:,:,1],f[:,:,2];
        gr = gg = gb = np.copy(g);

        for x in range(sizeMN[0] - 1):
            for y in range(sizeMN[1] - 1):
                somaR = somaG = somaB = 0;
                for i in range(-x1,x1+1):
                    for j in range(-y1,y1+1):
                        fx = x - i;
                        if(fx < 0):
                            fx = 0;
                        fy = y - j;
                        if(fy >= sizeMN[0]):
                            fy = sizeMN[0] - 1;
                            
                        somaR += mask[i,j]*fr[fx,fy];
                        somaG += mask[i,j]*fg[fx,fy];
                        somaB += mask[i,j]*fb[fx,fy];
                gr[x,y] = somaR;
                gg[x,y] = somaG;
                gb[x,y] = somaB;
        g[:,:,0], g[:,:,1], g[:,:,2] = gr, gg, gb; 

    return Image.fromarray(np.uint8(g));

#função blur
def blur(img):
    msk = maskBlur();
    return convolve(img,msk);


#função erode
def erode(img, mask):
    mask = np.asarray(mask);
    
    x1, y1 = mask.shape;
    x1 /= 2;
    y1 /= 2;
    f = np.asarray(img);
    sizeMN = size(img);
    g = np.zeros(f.shape);

    if(img.mode == "L"):
        for x in range(sizeMN[0] - 1):
            for y in range(sizeMN[1] - 1):                
                menor = 9999999;
                for i in range(-x1,x1+1):
                    for j in range(-y1,y1+1):
                        fx = x - i;
                        if(fx < 0):
                            fx = 0;
                        fy = y - j;
                        if(fy >= sizeMN[0]):
                            fy = sizeMN[0] - 1;
                            
                        if(mask[i,j] != 0 and f[fx,fy] < menor):
                            menor = f[fx,fy];
                            
                g[x,y] = menor;
    else:
        fr, fg, fb = f[:,:,0],f[:,:,1],f[:,:,2];
        gr = gg = gb = np.copy(g);

        for x in range(sizeMN[0] - 1):
            for y in range(sizeMN[1] - 1):
                menorR = menorG = menorB = 9999999;
                for i in range(-x1,x1+1):
                    for j in range(-y1,y1+1):
                        fx = x - i;
                        if(fx < 0):
                            fx = 0;
                        fy = y - j;
                        if(fy >= sizeMN[0]):
                            fy = sizeMN[0] - 1;
                            
                        if(mask[i,j] != 0 and fr[fx,fy] < menorR):
                            menorR = fr[fx,fy];
                        if(mask[i,j] != 0 and fg[fx,fy] < menorG):
                            menorG = fg[fx,fy];
                        if(mask[i,j] != 0 and fb[fx,fy] < menorB):
                            menorB = fb[fx,fy];
                            
                gr[x,y] = menorR;
                gg[x,y] = menorG;
                gb[x,y] = menorB;
        g[:,:,0], g[:,:,1], g[:,:,2] = gr, gg, gb; 

    return Image.fromarray(np.uint8(g));

#função dilate
def dilate(img, mask):
    mask = np.asarray(mask);
    
    x1, y1 = mask.shape;
    x1 /= 2;
    y1 /= 2;
    f = np.asarray(img);
    sizeMN = size(img);
    g = np.zeros(f.shape);

    if(img.mode == "L"):
        for x in range(sizeMN[0] - 1):
            for y in range(sizeMN[1] - 1):                
                maior = 0;
                for i in range(-x1,x1+1):
                    for j in range(-y1,y1+1):
                        fx = x - i;
                        if(fx < 0):
                            fx = 0;
                        fy = y - j;
                        if(fy >= sizeMN[0]):
                            fy = sizeMN[0] - 1;
                            
                        if(mask[i,j] != 0 and f[fx,fy] > maior):
                            maior = f[fx,fy];
                            
                g[x,y] = maior;
    else:
        fr, fg, fb = f[:,:,0],f[:,:,1],f[:,:,2];
        gr = gg = gb = np.copy(g);

        for x in range(sizeMN[0] - 1):
            for y in range(sizeMN[1] - 1):
                maiorR = maiorG = maiorB = 0;
                for i in range(-x1,x1+1):
                    for j in range(-y1,y1+1):
                        fx = x - i;
                        if(fx < 0):
                            fx = 0;
                        fy = y - j;
                        if(fy >= sizeMN[0]):
                            fy = sizeMN[0] - 1;
                            
                        if(mask[i,j] != 0 and fr[fx,fy] > maiorR):
                            maiorR = fr[fx,fy];
                        if(mask[i,j] != 0 and fg[fx,fy] > maiorG):
                            maiorG = fg[fx,fy];
                        if(mask[i,j] != 0 and fb[fx,fy] > maiorB):
                            maiorB = fb[fx,fy];
                            
                gr[x,y] = maiorR;
                gg[x,y] = maiorG;
                gb[x,y] = maiorB;
        g[:,:,0], g[:,:,1], g[:,:,2] = gr, gg, gb; 

    return Image.fromarray(np.uint8(g));
