# AWR1642 初始設置

由於 Project 基於 xWR16xx 系列的 demo ，因此只需使用 UniFlash 將 demo 燒入到板子即可。

### 安裝 mmWave SDK

前往 [MMWAVE-SDK](https://www.ti.com/tool/MMWAVE-SDK?utm_source=google&utm_medium=cpc&utm_campaign=epd-rap-null-58700008050676524_mmwave_sdk_userguide_rsa-cpc-evm-google-tw_int&utm_content=mmwave_sdk_userguide&ds_k=mmwave+sdk&DCM=yes&gad_source=1&gclid=CjwKCAiAzJOtBhALEiwAtwj8tmEo_bw24-bBta8tGb3DrseRJvbKeYJTs0L-_LuGjBFWNRtrIAYkSRoCXk0QAvD_BwE&gclsrc=aw.ds#downloads) 下載最新版本 mmWave SDK 並安裝。

### 安裝 UniFlash

再來前往 [UNIFLASH](https://www.ti.com/tool/UNIFLASH?utm_source=google&utm_medium=cpc&utm_campaign=epd-der-null-58700007985254383_uniflash_rsa-cpc-evm-google-tw_int&utm_content=uniflash&ds_k=uniflash&gad_source=1&gclid=CjwKCAiAzJOtBhALEiwAtwj8tiYFbuCL1dZcrLkB-aei0nsHlAaECXLliQeWQ7kmXszZYVdWWNuzzxoCZkMQAvD_BwE&gclsrc=aw.ds#downloads) 下載最新版本 UniFlash 並安裝。

### 設置 AWR1642BOOST

將 SOP0 與 SOP2 Close 並重新送電以設置成 Flash Mode 後，再打開 UniFlash 開啟 AWR1642BOOST

![image](https://gist.github.com/assets/98001197/9dd2eba5-bc8f-45d3-aacb-42d94c2780e3)

選擇 mmWave SDK 所提供 Demo 的 Flash Image 後將板子重新送電，再點選 Load Image 直到顯示 `[SUCCESS] Program Load completed successfully.`

> bin 檔案路徑在 `C:\ti\mmwave_sdk_03_06_01_00-LTS\packages\ti\demo\xwr16xx\mmw\` 中

![image](https://gist.github.com/assets/98001197/7b2ce341-aed9-4590-8b8e-ccfcf7759ca3)

燒入成功後將 SOP0 Open 並重新送電，再打開 [mmWave Demo Visualizer](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/) 跟隨提示安裝瀏覽器插件及軟體，成功後即可看到此畫面。

![image](https://gist.github.com/assets/98001197/ed581421-bc31-4a11-a868-805925954323)

將 Setup Details 中的 Platform 選擇至 xWR16xx 並點選 `SEND CONFIG TO MMWAVE DEVICE`

![image](https://gist.github.com/assets/98001197/ee25e9b7-8f55-46ba-b9f8-4fff03d6cd3f)

將 Ports 選擇對應的 COM 後按下 OK，跳出後再按一次 `SEND CONFIG TO MMWAVE DEVICE` 即可將設置資料傳至板子上。
> 沒有看到 Ports 選單就將板子 USB 重插再嘗試一次

![image](https://gist.github.com/assets/98001197/56555a74-65d6-4872-9b2f-7fa05353a577)

最後點選 Plots 就可以看到即時的雷達資料了。

![image](https://gist.github.com/assets/98001197/0b7cba38-1b3f-4ca0-9dd0-9d4a25accc9c)
