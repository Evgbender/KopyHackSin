package com.kopybara.test1

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.Toast
import android.widget.VideoView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.IOException
//import com.chaquo.python.Python

class MainActivity : AppCompatActivity() {

    private val requestPermissionCode = 1
    private val pickVideo = 2
    private lateinit var videoUri: Uri
//    lateinit var model: LiteModelMovenetSingleposeLightningTfliteFloat164
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        //Python.start(this)
        val pickVideoButton = findViewById<Button>(R.id.pick_video_button)

        pickVideoButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                //val process = Runtime.getRuntime().exec(" python ball_tracking.py -v  t.mp4")
                openGallery()
            } else {
                requestPermission()
            }
        }
    }

    private fun requestPermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            )
        ) {
            Toast.makeText(
                this,
                "Permission needed to access videos",
                Toast.LENGTH_LONG
            ).show()
        }
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
            requestPermissionCode
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == requestPermissionCode) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openGallery()
            } else {
                Toast.makeText(
                    this,
                    "Permission denied to access videos",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun openGallery() {
        val galleryIntent = Intent(
            Intent.ACTION_PICK,
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        )
        startActivityForResult(galleryIntent, pickVideo)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == pickVideo && resultCode == RESULT_OK) {
            val videoUri = data?.data
            videoUri?.let {
                this.videoUri = it
                processPlayVideoFromUri(it)
            }

// Делайте нужные операции с выбранным видео
        }
    }
    private fun processPlayVideoFromUri(uri: Uri) {
        try {
            val videoView = findViewById<VideoView>(R.id.video_view)


//            val model = LiteModelMovenetSingleposeLightningTfliteFloat164.newInstance(context)
//
//// Creates inputs for reference.
//            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.UINT8)
//            inputFeature0.loadBuffer(byteBuffer)
//
//// Runs model inference and gets result.
//            val outputs = model.process(inputFeature0)
//            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//// Releases model resources if no longer used.
//            model.close()


            videoView.setVideoURI(uri)
            videoView.start()
        } catch (e: IOException) {
            e.printStackTrace()
            Toast.makeText(this, "Failed to play video", Toast.LENGTH_SHORT).show()
        }
    }
}

