import av

container = av.open("D:/Dropbox/Movies/Xxx/Big Titty Trans Bimbo Gets Her Tight Asshole Fucked Hard - Britt.mp4")

index = 0
for frame in container.decode(video=0):
    index += 1
    if index %100 == 0:
        frame.to_image().save('frames/frame-%05d.jpg' % index)