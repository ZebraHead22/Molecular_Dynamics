import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import openpyxl


destination_path = "."

spec_path = "data_spec"
dip_path = "data_dip"


def print_spectrum(filename):
    base = os.path.basename(filename)
    general_name = os.path.splitext(base)[0]

    print(general_name)
    df = pd.read_csv(os.path.join(spec_path, filename), header=None, sep=" ")
    # df = pd.read_csv(os.path.join(spec_path,filename), header=None, sep="	")

    print(df.head())
    ax = df.plot(x=0, y=1)

    # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label.set_fontsize(12)

    ax.set_xlabel('Wave number, $cm^{-1}$')
    ax.set_ylabel('Spectral density, a.u.)')

    # ax.set_ylabel('Spectral density (a.u.)', fontsize=14)
    ax.get_legend().remove()
    plt.grid(linewidth=0.2)
    plt.subplots_adjust(left=0.15, bottom=0.1)
    figure = plt.gcf()
    # figure.set_size_inches(10, 5)

    plt.savefig(os.path.join(destination_path, 'jpg',
                general_name + '.jpg'), dpi=800)
    # plt.savefig(os.path.join(destination_path, 'eps', general_name + '.eps'), format='eps')


def print_compare_spectrum(filenames):
    filenames.sort()
    base = os.path.basename(filenames[0])
    general_name = os.path.splitext(base)[0] + '_compare'
    print(filenames)
    for spec in filenames:
        df = pd.read_csv(os.path.join(spec_path, spec), header=None, sep=" ")
        print(df.head())
        lable = spec.split("-")[-1].split(".")[0]
        plt.plot(np.array(df[0].tolist()), np.array(
            df[1].tolist()), label=lable)

    plt.xlabel('Wave number, $cm^{-1}$')
    plt.ylabel('Spectral density, a.u.')
    plt.subplots_adjust(left=0.15, bottom=0.1)
    plt.legend()
    plt.grid(linewidth=0.2)

    plt.savefig(os.path.join(destination_path, 'jpg',
                             general_name + '.jpg'), dpi=600)
# plt.savefig(os.path.join(destination_path, 'eps', general_name + '.eps'), format='eps')


def print_summ_spectrum(filenames):
    base = os.path.basename(filenames[0])
    general_name = os.path.splitext(base)[0] + '_summ'
    print(general_name)
    avgDataFrame = pd.read_csv(os.path.join(
        spec_path, filenames[0]), header=None, sep=" ")[1]
    dataList = pd.DataFrame({"0": avgDataFrame})

    for i in range(1, len(filenames)):
        df = pd.read_csv(os.path.join(
            spec_path, filenames[i]), header=None, sep=" ")
        print(i, filenames[i], 'before data =', avgDataFrame[22747])
        print(i, 'add data =', df[1][22747])
        avgDataFrame = avgDataFrame.add(df[1])
        print(i, 'after data =', avgDataFrame[22747])
        dataList[str(i)] = df[1]

    avgDataFrame = avgDataFrame / len(filenames)

    std = np.array(dataList.std(axis=1).tolist())

    for j in range(0, len(std)):
        if (j % 100 != 0):
            std[j] = 0

    df[1] = avgDataFrame
    print(df)
    df.to_excel(general_name + '.xlsx')

    plt.plot(np.array(df[0].tolist()), avgDataFrame)
    # plt.errorbar(np.array(df[0].tolist()), avgDataFrame, yerr=std, ecolor='black', capsize=3, elinewidth=1, markeredgewidth=1)
    plt.xlabel('Wave number, $cm^{-1}$')
    plt.ylabel('Spectral density, a.u.')
    plt.grid(linewidth=0.2)

    plt.savefig(os.path.join(destination_path, 'jpg',
                general_name + '.jpg'), dpi=600)
    plt.savefig(os.path.join(destination_path, 'eps',
                general_name + '.eps'), format='eps')

# def print_spectrum(filename):
#     base = os.path.basename(filename)
#     general_name = os.path.splitext(base)[0]

#     print(general_name)
#     df = pd.read_csv(os.path.join(spec_path,filename), header=None, sep=" ")
#     # df = pd.read_csv(os.path.join(spec_path,filename), header=None, sep="	")

#     print(df.head())
#     ax = df.plot(x=0, y=1,  kind="scatter")

#     ax.set_xlabel('Time (fs)')
#     ax.set_ylabel('E (kcal/mol×Å×e)')
#     # ax.get_legend().remove()
#     plt.grid(linewidth=0.2)
#     #ax.set_title('a sine wave')

#     plt.savefig(os.path.join(destination_path, 'jpg', general_name + '.jpg'), dpi=600)
#     plt.savefig(os.path.join(destination_path, 'eps', general_name + '.eps'), format='eps')
# #plt.show()


def print_dip(filename):
    base = os.path.basename(filename)
    general_name = os.path.splitext(base)[0]
    print(general_name)
    mul = int(general_name.split("_")[1])

    print(mul)
    # df = pd.read_csv(os.path.join(dip_path,filename), sep="\t")
    df = pd.read_csv(os.path.join(dip_path, filename), sep="  ")

    df["frame"] = df["frame"] * mul
    print(df.head())
    ax = df.plot(x="frame", y=["dip_x", "dip_y", "dip_z", "|dip|"])

    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Dipole moment (D)')
    # ax.get_legend().remove()
    plt.grid(linewidth=0.2)
    plt.legend(bbox_to_anchor=(1, 1))

    figure = plt.gcf()
    figure.set_size_inches(10, 2)
    # f = plt.figure()
    # f.set_figwidth(4)
    # f.set_figheight(1)

    plt.savefig(os.path.join(destination_path, 'jpg',
                general_name + '.jpg'), dpi=600)
    # plt.savefig(os.path.join(destination_path, 'eps', general_name + '.eps'), format='eps')
    # plt.show()


spec_filenames = next(os.walk(spec_path), (None, None, []))[2]
# print_compare_spectrum(spec_filenames)
for spec in spec_filenames:
    print_spectrum(spec)

spec_list = os.listdir(spec_path)
# print_compare_spectrum(spec_list)
# print_summ_spectrum(spec_list)

dip_filenames = next(os.walk(dip_path), (None, None, []))[2]
for dip in dip_filenames:
    print_dip(dip)
