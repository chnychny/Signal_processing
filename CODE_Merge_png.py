from PIL import Image,ImageDraw, ImageFont
import os
font_size = 15
font = ImageFont.truetype("arial.ttf", font_size) # arial.ttf 글씨체, font_size=15

datapath = '\\\\147.47.239.143\\SHRM-Personal\\Personal_Drive\\박찬희\\RESEARCH_2022\\1. PYCODE\\Motor_Signal_Analysis\\'
Figsavedir = datapath+'figure_IPM109steady\\'

Figsavedir_new = datapath+'figure_IPM109steady_combine\\'
Figsavemode = True
resize_img_size = (200,200)
time_img_size = (1000,600)
# time_series plot 이미지로 정렬
term = "_analytic_plane"
total_img_list = os.listdir(Figsavedir)
n_i_list = []
for t_list in total_img_list:
    if t_list.find(term) is -1 and t_list.find('.png.png') is not -1: # 없으면
        n_i_list.append(t_list)

print(n_i_list)
for i in range(len(n_i_list)):
    # t_fig_name = figname1 + 'ni' + str(n_i) + '.png' _name = '_analytic_plane' + str(pp) + ' ' + figname1 + 'ni' + str(n_i) + '.png'
    plane_fig_list=[]
    for img_list in total_img_list:
        if img_list.find('fig_scale+resamp_analytic_plane') is not -1 and img_list.find(n_i_list[i]) is not -1:
            plane_fig_list.append(img_list)
            if img_list.find('_analytic_plane0') is not -1:
                image1 = Image.open(Figsavedir+img_list)
            elif img_list.find('_analytic_plane1') is not -1:
                image2 = Image.open(Figsavedir + img_list)
            elif img_list.find('_analytic_plane2') is not -1:
                image3 = Image.open(Figsavedir + img_list)
        if img_list.find('fig_original_n_analytic_plane') is not -1 and img_list.find(n_i_list[i]) is not -1:
            if img_list.find('_analytic_plane0') is not -1:
                image4 = Image.open(Figsavedir+img_list)
            elif img_list.find('_analytic_plane1') is not -1:
                image5 = Image.open(Figsavedir + img_list)
            elif img_list.find('_analytic_plane2') is not -1:
                image6 = Image.open(Figsavedir + img_list)
    if not len(plane_fig_list)==0: # 빈 리스트가 아니면
        print(str(i)+'/'+str(len(n_i_list))+': '+n_i_list[i][:-8])
        image0 = Image.open(Figsavedir+n_i_list[i])
        image0 = image0.resize(time_img_size)
        image1 = image1.resize(resize_img_size)
        image2 = image2.resize(resize_img_size)
        image3 = image3.resize(resize_img_size)
        image4 = image4.resize(resize_img_size)
        image5 = image5.resize(resize_img_size)
        image6 = image6.resize(resize_img_size)
        image0_size = image0.size
        image1_size = image1.size
        image2_size = image2.size

        new_image = Image.new('RGB', (image0_size[0]+2*image1_size[0], 3*image1_size[1]), (250, 250, 250))
        new_image.paste(image0, (0, 0))
        new_image.paste(image4, (image0_size[0], 0))
        new_image.paste(image5, (image0_size[0], image1_size[1]))
        new_image.paste(image6, (image0_size[0], image1_size[1]+image2_size[1]))
        new_image.paste(image1, (image0_size[0]+image1_size[0], 0))
        new_image.paste(image2, (image0_size[0]+image1_size[0], image1_size[1]))
        new_image.paste(image3, (image0_size[0]+image1_size[0], image1_size[1] + image2_size[1]))
        draw = ImageDraw.Draw(new_image)
        draw.text((int((image0_size[0]+2*image1_size[0])//5),25),n_i_list[i][:-8]+" t-series, raw, scale+resamp", fill=(0, 0, 255),font=font)

        # new_image.show()

        if Figsavemode:
            new_image.save(Figsavedir_new+"Total_"+n_i_list[i][:-8]+".jpg", "JPEG")
        # new_image.close()
