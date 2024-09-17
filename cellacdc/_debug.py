import inspect

import pandas as pd

from . import printl

def print_all_callers():
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe, 2)
    outerframes_format = '\n'
    for frame in outerframes:
        outerframes_format = f'{outerframes_format}  * {frame.function}\n'
    printl(outerframes_format)

def _debug_lineage_tree(guiWin):
    posData = guiWin.data[guiWin.pos_i]
    columns = set()	
    for frame_i in range(len(posData.allData_li)):
        acdc_df = posData.allData_li[frame_i]['acdc_df']
        if acdc_df is not None:
            columns.update(acdc_df.reset_index().columns)
    printl(f"Columns in acdc_df: {columns}")

    from pandasgui import show as pgshow
    if guiWin.lineage_tree is not None and guiWin.lineage_tree.lineage_list is not None:
        lin_tree_df = pd.DataFrame()
        for i, df in enumerate(guiWin.lineage_tree.lineage_list):
            df = df.copy()
            # df = df.reset_index()
            df["frame_i"] = i
            lin_tree_df = pd.concat([lin_tree_df, df])

        if not isinstance(lin_tree_df.index, pd.RangeIndex):
            lin_tree_df = lin_tree_df.reset_index()

        lin_tree_df = (lin_tree_df
                    .set_index(["frame_i", "Cell_ID"])
                    .sort_index()
                    )
        if "level_0" in lin_tree_df.columns:
            lin_tree_df=lin_tree_df.drop(columns="level_0")

    acdc_df = pd.DataFrame()
    posData = guiWin.data[guiWin.pos_i]
    df_li = [posData.allData_li[i]['acdc_df'] for i in range(len(posData.allData_li))]
    for i, df in enumerate(df_li):
        if df is None:
            continue
        df = df.copy()
        df = df.reset_index()
        df["frame_i"] = i
        acdc_df = pd.concat([acdc_df, df])

    acdc_df = (acdc_df
                .set_index(["frame_i", "Cell_ID"])
                .sort_index()
                )

    # for key, value in guiWin.lineage_tree.family_dict.items():
    if guiWin.lineage_tree is not None and guiWin.lineage_tree.lineage_list is not None:
        families = pd.DataFrame()
        for family in guiWin.lineage_tree.families:
            family_name = family[0][0]
            family_df = pd.DataFrame(family, columns=["Cell_ID", "generation_num_tree"])
            family_df["family_name"] = family_name
            family_df = family_df.set_index("family_name")
            families = pd.concat([families, family_df])
        if "level_0" in families.columns:
            families=families.drop(columns="level_0")

    # lin_tree_dict_df = (lin_tree_dict_df
    #     .set_index(["family_name", "frame_i", "Cell_ID"])
    #     .sort_index()
    #     )
    
    # for i, df in enumerate([acdc_df, lin_tree_df, families, lin_tree_dict_df]):
    #     printl(f"Columns: {df.columns} for df {i}" )
    #     if (df.columns == df.index.name).any():
    #         printl(f"Index name: {df.index.name} for df {i}!!!" )

    if "level_0" in acdc_df.columns:
        acdc_df=acdc_df.drop(columns="level_0")


    if guiWin.lineage_tree is not None and guiWin.lineage_tree.lineage_list is not None:
        pgshow(acdc_df, lin_tree_df, families)
    else:
        pgshow(acdc_df)

    printl(posData.tracked_lost_centroids)