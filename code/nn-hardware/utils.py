def toHex(num, bits):
    return hex((num + (1<<bits)) % (1<<bits))

def createTestImage(loader):
    import torch
    from fxpmath import Fxp

    test_file = open("test_image.txt", "w")
    with torch.no_grad():
        count = 1
        for data in loader:
            X, y = data
            pic = X.view(-1, 784)

            for i in pic:
                for j in i:
                    final = toHex(Fxp(j.item(), n_int=16, n_frac=16, signed=True).raw(), 32)
                    final = str(final).replace("0x", "")
                    if (len(final) < 8):
                        for k in range(8 - len(final)):
                            final = "0" + str(final)
                    test_file.write("{}\n".format(final))
            if count == 1:
                break