import gradio as gr
import numpy as np
import cv2
import image_function


with gr.Blocks() as page1:
    gr.Markdown("""# Histogram analysis and threshold operation""")
    with gr.Row():
        input_image=gr.Image()
        output_image=gr.Image()
    with gr.Row():
        output_image2=gr.Image()
        output_image3=gr.Image()
    with gr.Row():
        threshold= gr.Slider(minimum=0, maximum=255, value=127, label="Threshold")
        with gr.Row():
            btn0=gr.Button('grey image')
            btn1=gr.Button('histogram')
            btn2=gr.Button('threshold_manual')
            btn3=gr.Button('threshold-Ostu')
    btn0.click(fn=image_function.RGB2Grey,inputs=input_image,outputs=output_image)
    btn1.click(fn=image_function.histogram,inputs=output_image,outputs=output_image2)
    btn2.click(fn=image_function.threshold_manual,inputs=[output_image,threshold],outputs=output_image3)
    btn3.click(fn=image_function.threshold_Otsu,inputs=output_image,outputs=output_image3)
page1.title="Histogram analysis and threshold operation"

with gr.Blocks() as page2:
    gr.Markdown("""# Convolution and Image Filters""")
    with gr.Row():
        input_image=gr.Image()
        output_image=gr.Image()
    with gr.Row():
        with gr.Column():
            operator1=gr.Dropdown(["roberts",'prewitt',"sobel"],value='roberts',info='fillters for edge detection',label='edge detection')
            btn1=gr.Button("process")
        with gr.Column():
            operator2=gr.Dropdown(["gaussian","median"],value='gaussian',info='fillters for noise reduction',label='noise reduction')
            btn2=gr.Button("process")
    btn1.click(fn=image_function.convolutionAndImageFilters2,inputs=[input_image,operator1],outputs=output_image)
    btn2.click(fn=image_function.convolutionAndImageFilters,inputs=[input_image,operator2],outputs=output_image)
page2.title="Convolution and Image Filters"

with gr.Blocks() as page3:
    gr.Markdown("""# dilation,erosion,opening and closing operations in grayscale """)
    with gr.Row():
        input_image=gr.Image()
        output_image=gr.Image()
    with gr.Row():
        kernelsize=gr.Slider(minimum=1,maximum=20,value=3,label="kernel size",step=1)
        btn0=gr.Button('set output as input')
        btn1=gr.Button('dilation')
        btn2=gr.Button('erosion')
        btn3=gr.Button('opening')
        btn4=gr.Button('closing')
    btn0.click(fn=lambda x:x,inputs=output_image,outputs=input_image)
    btn1.click(fn=image_function.dilation_gray,inputs=[input_image,kernelsize],outputs=output_image)
    btn2.click(fn=image_function.erosion_gray,inputs=[input_image,kernelsize],outputs=output_image)
    btn3.click(fn=image_function.open_gray,inputs=[input_image,kernelsize],outputs=output_image)
    btn4.click(fn=image_function.close_gray,inputs=[input_image,kernelsize],outputs=output_image)
page3.title="dilation,erosion,opening and closing operations in grayscale"

with gr.Blocks() as page4:
    gr.Markdown("""# Morphological edge detection,reconstruction and gradient""")
    with gr.Row():
        input_image=gr.Image(tool='sketch')
        output_image=gr.Image()
    with gr.Row():
        btn0=gr.Button('Morphological edge detection')
        btn1=gr.Button("Morphological gradient")
    with gr.Row():
        btn2=gr.Button("conditional dilation in binary image")
        btn3=gr.Button("Gray scale reconstruction")
    btn0.click(fn=image_function.MorphologicalEdgeDetection,inputs=input_image,outputs=output_image)
    btn1.click(fn=image_function.MorphologiclGradient,inputs=input_image,outputs=output_image)
    btn2.click(fn=image_function.ConditionalDilation,inputs=input_image,outputs=output_image)
    btn3.click(fn=image_function.GrayScaleReconstruction,inputs=input_image,outputs=output_image)
page4.title="Morphological edge detection,reconstruction and gradient"


with gr.Blocks() as page5:
    gr.Markdown("""# dilation,erosion,opening and closing operations in binagry scale """)
    with gr.Row():
        input_image=gr.Image()
        output_image=gr.Image()
    with gr.Row():
        kernelsize=gr.Slider(minimum=1,maximum=20,value=3,label="kernel size",step=1)
        btn0=gr.Button('set output as input')
        btn1=gr.Button('dilation')
        btn2=gr.Button('erosion')
        btn3=gr.Button('opening')
        btn4=gr.Button('closing')
    btn0.click(fn=lambda x:x,inputs=output_image,outputs=input_image)
    btn1.click(fn=image_function.dilation_bin,inputs=[input_image,kernelsize],outputs=output_image)
    btn2.click(fn=image_function.erosion_bin,inputs=[input_image,kernelsize],outputs=output_image)
    btn3.click(fn=image_function.open_bin,inputs=[input_image,kernelsize],outputs=output_image)
    btn4.click(fn=image_function.close_bin,inputs=[input_image,kernelsize],outputs=output_image)
page5.title="dilation,erosion,opening and closing operations in binagry scale "

with gr.Blocks() as page6:
    gr.Markdown("""# Morphological distance transform,skeleton and skeleton restoration""")
    with gr.Row():
        input_image=gr.Image(label='input bin image')
        output_image=gr.Image(label='distance transform')
    with gr.Row():
        output_image2=gr.Image(label='skeleton')
        output_image3=gr.Image(label='img recovered from skeleton')
    with gr.Row():
        btn0=gr.Button('distance transform')
        btn1=gr.Button('generate skeleton')
        btn2=gr.Button('recovered from skeleton')

    btn0.click(fn=image_function.DistanceTransform,inputs=input_image,outputs=output_image)
    btn1.click(fn=image_function.skeleton,inputs=input_image,outputs=output_image2)
    btn2.click(fn=image_function.skeleton_recover,inputs=output_image2,outputs=output_image3)
page6.title="Morphological distance transform,skeleton and skeleton restoration"


from fastapi import FastAPI, Response



app = FastAPI()

@app.get('/')
async def root():
    html_content = """
    <html>
    <head>
        <title>Image Processing Homework</title>
    </head>
    <body>
        <h1>Image Processing Homework<</h1>
        <p>Author: 马浩吉</p>
        <p>Click the links below to navigate:</p>
        <ul>
            <li><a href="/page1">Page 1: Histogram analysis and threshold operation</a></li>
            <li><a href="/page2">Page 2: Convolution and Image Filters</a></li>
            <li><a href="/page3">Page 3: dilation,erosion,opening and closing operations in grayscale</a></li>
            <li><a href="/page4">Page 4: Morphological edge detection,reconstruction and gradient</a></li>
            <li><a href="/page5">Page 5: ilation,erosion,opening and closing operations in binagry scale</a></li>
            <li><a href="/page6">Page 6: Morphological distance transform,skeleton and skeleton restoration</a></li>

        </ul>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

app = gr.mount_gradio_app(app, page1, path='/page1')
app = gr.mount_gradio_app(app, page2, path='/page2')
app = gr.mount_gradio_app(app, page3, path='/page3')
app = gr.mount_gradio_app(app, page4, path='/page4')
app = gr.mount_gradio_app(app, page5, path='/page5')
app = gr.mount_gradio_app(app, page6, path='/page6')

import uvicorn
uvicorn.run(app,port=1234)