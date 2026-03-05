package global

// 这里定义的常量，一般是具有错误代码+错误说明组成，一般用于接口返回
type Alg struct {
	Type string
	Alg  int
}

const (
	// 进程被结束
	ProcessKilled string = "收到信号，进程被结束"

	//校验器相关
	ValidatorParamsCheckFailCode int = 400300

	//服务器代码发生错误
	ServerOccurredErrorCode int = -500100

	// token相关
	LoginSuccess          int = 200100 //token有效
	JwtTokenInvalid       int = 400100 //无效的token
	JwtTokenExpired       int = 400101 //过期的token
	JwtTokenFormatErrCode int = 400102 //提交的 token 格式错误
	LoginInfoInavlid      int = 400103 //没有提交登录信息
	// CURD 常用业务状态码
	CurdStatusOkCode         int    = 200
	CurdStatusOkMsg          string = "Success"
	CurdCreatFailCode        int    = -400200
	CurdCreatFailMsg         string = "新增失败"
	CurdUpdateFailCode       int    = -400201
	CurdUpdateFailMsg        string = "更新失败"
	CurdDeleteFailCode       int    = -400202
	CurdDeleteFailMsg        string = "删除失败"
	CurdSelectFailCode       int    = -400203
	CurdSelectFailMsg        string = "查询无数据"
	CurdRegisterFailCode     int    = -400204
	CurdRegisterFailMsg      string = "注册失败"
	CurdLoginFailCode        int    = -400205
	CurdLoginFailMsg         string = "登录失败"
	CurdRefreshTokenFailCode int    = -400206
	CurdRefreshTokenFailMsg  string = "刷新Token失败"

	//文件上传
	FilesUploadFailCode            int    = -400250
	FilesUploadFailMsg             string = "文件上传失败, 获取上传文件发生错误!"
	FilesUploadMoreThanMaxSizeCode int    = -400251
	FilesUploadMoreThanMaxSizeMsg  string = "长传文件超过系统设定的最大值,系统允许的最大值："
	FilesUploadMimeTypeFailCode    int    = -400252
	FilesUploadMimeTypeFailMsg     string = "文件mime类型不允许"
	FilesUploadIsEmpty             string = "不允许上传空文件"

	//验证码
	CaptchaGetParamsInvalidMsg    string = "获取验证码：提交的验证码参数无效,请检查验证码ID以及文件名后缀是否完整"
	CaptchaGetParamsInvalidCode   int    = -400350
	CaptchaCheckParamsInvalidMsg  string = "校验验证码：提交的参数无效，请检查 【验证码ID、验证码值】 提交时的键名是否与配置项一致"
	CaptchaCheckParamsInvalidCode int    = -400351
	CaptchaCheckOkMsg             string = "验证码校验通过"
	CaptchaCheckFailCode          int    = -400355
	CaptchaCheckFailMsg           string = "验证码校验失败"
	ValidatorPrefix               string = "Form_Validator_" // 表单验证器前缀
	ConfigKeyPrefix               string = "ConfigKey_"

	//  SnowFlake 雪花算法
	StartTimeStamp = int64(1483228800000)              //开始时间截 (2017-01-01)
	MachineIdBits  = uint(10)                          //机器id所占的位数
	SequenceBits   = uint(12)                          //序列所占的位数
	MachineIdMax   = int64(-1 ^ (-1 << MachineIdBits)) //支持的最大机器id数量
	SequenceMask   = int64(-1 ^ (-1 << SequenceBits))  //
	MachineIdShift = SequenceBits                      //机器id左移位数
	TimestampShift = SequenceBits + MachineIdBits      //时间戳左移位数

	//token类型
	AccessToken  = 0
	RefreshToken = 1

	// WebAuthn 加密算法标识符
	AlgES256 = -7   // ECDSA with SHA-256
	AlgES384 = -35  // ECDSA with SHA-384
	AlgES512 = -36  // ECDSA with SHA-512
	AlgRS256 = -257 // RSASSA-PKCS1-v1_5 with SHA-256
	AlgEdDSA = -8   // EdDSA

	//服务器环境
	// 1.生产环境 2.开发环境 3.预发布环境 4.单元测试环境
	production  string = "production"
	development string = "development"
	staging     string = "staging"
	test        string = "test"

	//redis相关
	Redis_UserID_seesion string = ""
)

// 变量
var BasePath string

var PubKeyCredParams = []Alg{
	{"public-key", AlgES256},
	{"public-key", AlgES384},
	{"public-key", AlgES512},
	{"public-key", AlgRS256},
	{"public-key", AlgEdDSA},
}
